#!/usr/bin/env bash
# =============================================================================
# sweep_study2_segments.sh
#
# Study 2: Segment sensitivity check (supplemental)
#   - Fixed: n_qubits = 8
#   - Fixed: total_shots = 50000  (replace with your N* from Study 1 if different)
#   - Segments: all powers of two from 2 up to 2^8 = 256
#   - 40 reps per configuration
#
# Produces a separate CSV for the segment sensitivity plot.
# =============================================================================
set -euo pipefail

DATA="/Users/navalverde/work/SF/qae/risk_qae/examples/policy_data.npy"
TS_RUN="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="runs/study2_segments_${TS_RUN}.csv"
RUN_ID="study2_segments_${TS_RUN}"

REPS=40
NQ=8
TOTAL_SHOTS=60000   # <-- replace with N* once you have it from Study 1
ALPHAS="0.95,0.99"
GROVER_POWERS="0,1,2,4,8"
SHOTS_PER_CALL=2000
MAX_CALLS=100
SEED=7

echo "============================================"
echo "Study 2: Segment sensitivity"
echo "Run ID : $RUN_ID"
echo "Qubits : $NQ  (fixed)"
echo "Shots  : $TOTAL_SHOTS  (fixed)"
echo "Output : $OUT"
echo "============================================"

# All powers of two from 2^1=2 up to 2^8=256
for SEGS in 2 4 8 16 32 64 128 256; do
  echo "  SEGS=$SEGS"
  python run_one.py \
    --data      "$DATA" \
    --out       "$OUT" \
    --total_shots "$TOTAL_SHOTS" \
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

echo ""
echo "Done. Output -> $OUT"
echo "Next: python analyze_runs.py  (set CSV_FILE and STUDY='segments' in USER SETTINGS)"
