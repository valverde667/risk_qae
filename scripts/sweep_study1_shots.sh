#!/usr/bin/env bash
# =============================================================================
# sweep_study1_shots.sh
#
# Study 1: Shot sweep
#   - Qubit counts: 6, 8, 10
#   - Segments: maximum viable (2^n_qubits) for each qubit count
#   - Shots: 20k, 30k, 40k, 50k, 60k, 70k, 80k
#   - 40 reps per configuration
#
# All results land in a single CSV for analyze_runs.py.
# =============================================================================
set -euo pipefail

DATA="/Users/navalverde/work/SF/qae/risk_qae/examples/policy_data.npy"
TS_RUN="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="runs/study1_shots_${TS_RUN}.csv"
RUN_ID="study1_shots_${TS_RUN}"

REPS=40
ALPHAS="0.95,0.99"
GROVER_POWERS="0,1,2,4,8"
SHOTS_PER_CALL=2000
MAX_CALLS=100
SEED=7

echo "============================================"
echo "Study 1: Shot sweep"
echo "Run ID : $RUN_ID"
echo "Output : $OUT"
echo "============================================"

# Qubit counts and their maximum viable segment counts (2^n)
# Written as a case statement for bash 3 compatibility (macOS default shell)
max_segs_for() {
  case "$1" in
    6)  echo 64   ;;
    8)  echo 64  ;;
    10) echo 64 ;;
    *)  echo $((2 ** $1)) ;;
  esac
}

for NQ in 6 8 10; do
  SEGS="$(max_segs_for $NQ)"
  for TS in 20000 30000 40000 50000 60000 70000 80000; do
    echo "  NQ=$NQ  SEGS=$SEGS  SHOTS=$TS"
    python run_one.py \
      --data      "$DATA" \
      --out       "$OUT" \
      --total_shots "$TS" \
      --n_index_qubits "$NQ" \
      --n_segments "$SEGS" \
      --reps      "$REPS" \
      --run_id    "$RUN_ID" \
      --alphas    "$ALPHAS" \
      --grover_powers "$GROVER_POWERS" \
      --shots_per_call "$SHOTS_PER_CALL" \
      --max_calls "$MAX_CALLS" \
      --seed      "$SEED"
  done
done

echo ""
echo "Done. Output -> $OUT"
echo "Next: python analyze_runs.py  (set CSV_FILE and STUDY in USER SETTINGS)"
