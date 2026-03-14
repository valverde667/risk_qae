#!/usr/bin/env bash
set -euo pipefail

DATA="/Users/navalverde/work/SF/qae/risk_qae/examples/mc_losses_gamma_model.npy"
TS_RUN="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="runs/grid_${TS_RUN}.csv"

REPS=40

for NQ in 6 7 8 9 10; do
  for TS in 2000 5000 10000 20000 40000 80000; do
    python run_one.py \
      --data "$DATA" \
      --out "$OUT" \
      --total_shots "$TS" \
      --n_index_qubits "$NQ" \
      --reps "$REPS" \
      --run_id "grid_${TS_RUN}" \
      --alphas "0.95,0.99" \
      --grover_powers "0,1,2,4,8" \
      --shots_per_call 2000 \
      --max_calls 100 \
      --seed 7
  done
done

echo "Wrote -> $OUT"
