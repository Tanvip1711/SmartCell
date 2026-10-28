
import subprocess
import sys
from pathlib import Path

# Folder containing this script
SRC_DIR = Path(__file__).resolve().parent

# Run scripts in pipeline order
scripts = [
    "day1_test.py",
    "preprocessing.py",
    "segmentation.py",
    "contour_detection.py",
    "day5_evaluation.py"
]

print("=" * 50)
print("       SMARTCELL - COMPLETE PIPELINE")
print("=" * 50)

for script in scripts:
    script_path = SRC_DIR / script

    print(f"\nRunning: {script}")
    print("-" * 50)

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SRC_DIR.parent)
    )

    if result.returncode != 0:
        print(f"\nFAILED: {script}")
        print("Fix the error above before continuing.")
        sys.exit(result.returncode)

    print(f"COMPLETED: {script}")

print("\n" + "=" * 50)
print("SMARTCELL PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 50)
