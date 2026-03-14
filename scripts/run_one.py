from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from datetime import datetime

import numpy as np

from risk_qae import RiskQAEConfig, estimate_risk_measures
from risk_qae.config import (
    AEConfig,
    BackendConfig,
    BudgetConfig,
    DiscretizationConfig,
    ValueEncodingConfig,
)
from risk_qae.discretization.histogram import discretize_samples
from risk_qae.metrics.classical import classical_mean, classical_var, classical_tvar


def rel_err(est: float, ref: float) -> float:
    denom = max(abs(ref), 1e-12)
    return abs(est - ref) / denom


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_csv_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_csv_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", type=str, required=True, help="Path to .npy file of samples"
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="CSV path to append results OR a directory",
    )

    ap.add_argument("--total_shots", type=int, required=True)
    ap.add_argument("--shots_per_call", type=int, default=2000)
    ap.add_argument("--max_calls", type=int, default=100)

    ap.add_argument("--n_index_qubits", type=int, default=8)
    ap.add_argument("--alphas", type=str, default="0.95,0.99")

    ap.add_argument("--ae_method", type=str, default="grover_mle")
    ap.add_argument("--grover_powers", type=str, default="0,1,2,4,8")
    ap.add_argument("--grid_size", type=int, default=2000)
    ap.add_argument("--alpha_conf", type=float, default=0.05)
    ap.add_argument("--epsilon_target", type=float, default=None)

    ap.add_argument("--value_encoding", type=str, default="piecewise_prefix")
    ap.add_argument("--n_segments", type=int, default=16)

    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument(
        "--reps", type=int, default=1, help="Number of repeated runs at same settings"
    )
    ap.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional label to group runs (defaults to UTC timestamp)",
    )

    args = ap.parse_args()

    # ---- load data ----
    samples = np.load(args.data).astype(float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise SystemExit("No finite samples found in data file.")

    alphas = parse_csv_floats(args.alphas)
    powers = parse_csv_ints(args.grover_powers)

    # ---- prepare output path ----
    out_path = Path(args.out)
    ensure_parent(out_path)

    if out_path.suffix.lower() != ".csv":
        out_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_path = out_path / f"risk_qae_run_{ts}.csv"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- run id ----
    run_id = args.run_id
    if run_id is None or str(run_id).strip() == "":
        run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # ---- config template ----
    cfg_template = RiskQAEConfig(
        backend=BackendConfig(provider="aer"),
        discretization=DiscretizationConfig(n_index_qubits=args.n_index_qubits),
        budget=BudgetConfig(
            total_shots=args.total_shots,
            shots_per_call=args.shots_per_call,
            max_circuit_calls=args.max_calls,
            seed=args.seed,
        ),
        value_encoding=ValueEncodingConfig(
            method=args.value_encoding, n_segments=args.n_segments
        ),
        ae=AEConfig(
            method=args.ae_method,
            grover_mle_grid_size=args.grid_size,
            grover_powers=powers,
            alpha_confidence=args.alpha_conf,
            epsilon_target=args.epsilon_target,
        ),
    )

    # ---- classical reference (fixed for all reps) ----
    dist = discretize_samples(samples, cfg_template.discretization)
    cmean = classical_mean(dist).mean
    cvar = {a: classical_var(dist, a).var for a in alphas}
    ctvar = {a: classical_tvar(dist, a).tvar for a in alphas}

    # ---- execute reps ----
    all_rows: list[dict] = []

    for rep in range(int(args.reps)):
        rep_seed = int(args.seed) + rep

        cfg = RiskQAEConfig(
            backend=cfg_template.backend,
            discretization=cfg_template.discretization,
            bounds=cfg_template.bounds,
            value_encoding=cfg_template.value_encoding,
            ae=cfg_template.ae,
            var_search=cfg_template.var_search,
            diagnostics=cfg_template.diagnostics,
            budget=type(cfg_template.budget)(
                total_shots=cfg_template.budget.total_shots,
                shots_per_call=cfg_template.budget.shots_per_call,
                max_circuit_calls=cfg_template.budget.max_circuit_calls,
                allocation=cfg_template.budget.allocation,
                seed=rep_seed,
            ),
        )

        # ---- timed quantum run ----
        t0 = time.perf_counter()
        try:
            res = estimate_risk_measures(
                {"samples": samples}, alphas=alphas, config=cfg
            )
        except Exception as exc:
            print(f"  [WARN] rep={rep} seed={rep_seed} failed: {exc}")
            continue
        runtime_s = time.perf_counter() - t0

        row_base = {
            "ts": datetime.utcnow().isoformat(),
            "run_id": str(run_id),
            "rep": int(rep),
            "seed": int(rep_seed),
            "data": str(Path(args.data).name),
            "n_samples": int(samples.size),
            "n_index_qubits": int(args.n_index_qubits),
            "total_shots_budget": int(args.total_shots),
            "shots_per_call": int(args.shots_per_call),
            "max_calls": int(args.max_calls),
            "ae_method": args.ae_method,
            "grover_powers": ",".join(map(str, powers)),
            "grid_size": int(args.grid_size),
            "alpha_conf": float(args.alpha_conf),
            "epsilon_target": (
                args.epsilon_target
                if args.epsilon_target is None
                else float(args.epsilon_target)
            ),
            "value_encoding": args.value_encoding,
            "n_segments": int(args.n_segments),
            "bounds_min": float(dist.bounds[0]),
            "bounds_max": float(dist.bounds[1]),
            # ---- timing (new) ----
            "runtime_s": float(runtime_s),
            # ---- mean ----
            "mean_classical": float(cmean),
            "mean_quantum": float(res.mean.mean),
            "mean_rel_err": float(rel_err(res.mean.mean, cmean)),
            "mean_shots_used": int(res.mean.shots_used),
            "mean_circuits_run": int(res.mean.circuits_run),
            "total_shots_used": int(res.total_shots_used),
            "total_circuits_run": int(res.total_circuits_run),
        }

        for a in alphas:
            vr = res.var[a]
            tr = res.tvar[a]

            all_rows.append(
                {
                    **row_base,
                    "alpha": float(a),
                    # VaR
                    "var_classical": float(cvar[a]),
                    "var_quantum": float(vr.var),
                    "var_rel_err": float(rel_err(vr.var, cvar[a])),
                    "var_bin_index": int(vr.var_bin_index),
                    "var_bracket_lo": float(vr.bracket[0]),
                    "var_bracket_hi": float(vr.bracket[1]),
                    "var_shots_used": int(vr.shots_used),
                    "var_circuits_run": int(vr.circuits_run),
                    "var_bisection_iters": int(
                        vr.diagnostics.get("iters", -1) if vr.diagnostics else -1
                    ),
                    # TVaR
                    "tvar_classical": float(ctvar[a]),
                    "tvar_quantum": float(tr.tvar),
                    "tvar_rel_err": float(rel_err(tr.tvar, ctvar[a])),
                    "tvar_tail_prob": float(tr.tail_prob),
                    "tvar_tail_mean_numer": float(tr.tail_mean_numerator),
                    "tvar_tail_prob_shots_used": int(
                        getattr(tr, "tail_prob_shots_used", -1)
                    ),
                    "tvar_tail_prob_circuits_run": int(
                        getattr(tr, "tail_prob_circuits_run", -1)
                    ),
                    "tvar_tail_component_shots_used": int(
                        getattr(tr, "tail_component_shots_used", -1)
                    ),
                    "tvar_tail_component_circuits_run": int(
                        getattr(tr, "tail_component_circuits_run", -1)
                    ),
                    "tvar_shots_used": int(tr.shots_used),
                    "tvar_circuits_run": int(tr.circuits_run),
                }
            )

    if not all_rows:
        raise SystemExit("All reps failed — no rows to write.")

    # ---- write CSV ----
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        if write_header:
            w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"Wrote {len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
