#!/bin/bash
# Run only tet r5 tests to verify the PCG fix
# This avoids running all 51 night tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/tet-r5-fix-verification-$(date +%Y%m%d-%H%M%S)}"

echo "=========================================="
echo "Tet r5 PCG Fix Verification"
echo "=========================================="
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Run 3 production mode repeats (enough to verify the fix)
REPEATS=(1 2 3)

for rep in "${REPEATS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running tet-r5-production-rep$(printf "%02d" $rep)"
    echo "=========================================="

    REP_DIR="$OUTPUT_DIR/tet-r5-production-rep$(printf "%02d" $rep)"
    LOG_FILE="$OUTPUT_DIR/tet-r5-production-rep$(printf "%02d" $rep).log"

    cd "$WORKTREE_DIR"
    "$WORKTREE_DIR/.venv/bin/python" -m scripts.monolithic_reference.run_p2_8bc \
        --kind tet \
        --refinement 5 \
        --device cuda:0 \
        --mass-matrix reference \
        --pcg-mode production \
        --output "$REP_DIR" \
        2>&1 | tee "$LOG_FILE"

    if [ $? -eq 0 ]; then
        echo "✓ Rep $(printf "%02d" $rep) completed successfully"
    else
        echo "✗ Rep $(printf "%02d" $rep) failed"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "Analyzing Results"
echo "=========================================="

# Extract PCG statistics from all runs
python3 <<ANALYSIS
import json
import sys
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
results = []

for rep in [1, 2, 3]:
    json_file = output_dir / f"tet-r5-production-rep{rep:02d}" / "run.json"
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

    print(f"Rep {rep:02d}: Total PCG={total_pcg:,} | p95={pcg_p95} | Time={solver_time:.3f}s")

if not results:
    print("\n✗ No results found!")
    sys.exit(1)

print("\n" + "="*50)
print("Expected baseline values (from 8C analysis):")
print("  Total PCG iterations: ~262k-271k")
print("  PCG p95: 200")
print("  Solver time: ~117s")
print("")
print("Previous regression (before fix):")
print("  Total PCG iterations: ~313k-340k (+19-27%)")
print("  PCG p95: 200")
print("  Solver time: ~128s-131s (+9-12%)")
print("="*50)

avg_pcg = sum(r["total_pcg"] for r in results) / len(results)
avg_time = sum(r["solver_time"] for r in results) / len(results)
max_p95 = max(r["pcg_p95"] for r in results)

print(f"\nCurrent results (after fix):")
print(f"  Avg Total PCG: {avg_pcg:,.0f}")
print(f"  Max p95: {max_p95}")
print(f"  Avg Time: {avg_time:.3f}s")

# Check if fix is effective
baseline_pcg_upper = 271000
if avg_pcg <= baseline_pcg_upper * 1.05:  # Allow 5% variance
    print("\n✓ PCG iterations RECOVERED to baseline level!")
else:
    print(f"\n⚠ PCG iterations still elevated (>{baseline_pcg_upper*1.05:,.0f})")

if max_p95 <= 200:
    print("✓ PCG p95 gate PASSED!")
else:
    print(f"✗ PCG p95 gate FAILED (>{200})")

ANALYSIS

echo ""
echo "=========================================="
echo "Verification Complete"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
