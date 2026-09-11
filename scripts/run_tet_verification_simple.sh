#!/bin/bash
# Simplified tet r5 verification script

set -e

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/tet-fix-test-$(date +%Y%m%d-%H%M%S)}"
WORKTREE=/home/lightwheel/Desktop/newton/SamiulJ/worktrees/monolithic-integration

echo "=========================================="
echo "Tet r5 PCG Fix Verification"
echo "=========================================="
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

cd "$WORKTREE"

# Run 3 production mode repeats
for rep in 1 2 3; do
    echo ""
    echo "=========================================="
    echo "Running tet-r5-production-rep$(printf "%02d" $rep)"
    echo "=========================================="

    REP_DIR="$OUTPUT_DIR/rep$(printf "%02d" $rep)"
    LOG_FILE="$OUTPUT_DIR/rep$(printf "%02d" $rep).log"

    $WORKTREE/.venv/bin/python -m scripts.monolithic_reference.run_p2_8bc \
        --kind tet \
        --refinement 5 \
        --device cuda:0 \
        --mass-matrix reference \
        --pcg-mode production \
        --output "$REP_DIR" \
        2>&1 | tee "$LOG_FILE"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Rep $(printf "%02d" $rep) completed"
    else
        echo "✗ Rep $(printf "%02d" $rep) failed"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "Analyzing Results"
echo "=========================================="

# Analyze results
$WORKTREE/.venv/bin/python3 <<EOF
import json
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
results = []

for rep in [1, 2, 3]:
    json_file = output_dir / f"rep{rep:02d}" / "run.json"
    if not json_file.exists():
        print(f"✗ Missing: {json_file}")
        continue

    with open(json_file) as f:
        data = json.load(f)

    total_pcg = data.get("total_pcg_iterations", 0)
    pcg_p95 = data.get("pcg_p95", 0)
    solver_time = data.get("solver_seconds", 0)

    results.append({
        "rep": rep,
        "total_pcg": total_pcg,
        "pcg_p95": pcg_p95,
        "solver_time": solver_time
    })

    print(f"Rep {rep:02d}: PCG={total_pcg:,} | p95={pcg_p95} | Time={solver_time:.2f}s")

if not results:
    print("\n✗ No results found!")
    exit(1)

print("\n" + "="*60)
print("Baseline (8C before fix):  PCG ~262k-271k | Time ~117s")
print("Regression (8C with bug):  PCG ~313k-340k | Time ~128s-131s")
print("="*60)

avg_pcg = sum(r["total_pcg"] for r in results) / len(results)
avg_time = sum(r["solver_time"] for r in results) / len(results)
max_p95 = max(r["pcg_p95"] for r in results)

print(f"\nAfter fix (current): PCG {avg_pcg:,.0f} | Time {avg_time:.2f}s | p95 {max_p95}")

baseline_upper = 285000  # 271k + 5% tolerance
if avg_pcg <= baseline_upper:
    print("\n✓ PCG RECOVERED to baseline level!")
    recovery = True
else:
    print(f"\n✗ PCG still elevated (>{baseline_upper:,})")
    recovery = False

if max_p95 <= 200:
    print("✓ p95 gate PASSED!")
    gate = True
else:
    print(f"✗ p95 gate FAILED (>{200})")
    gate = False

exit(0 if (recovery and gate) else 1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ FIX VERIFIED - All tests passed!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "⚠ Issues detected - review results above"
    echo "=========================================="
    exit 1
fi
