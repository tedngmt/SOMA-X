"""Run the matched GraspXL package build with persistent disk logging."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
log = root / 'out/graspxl-matched-conversion.log'
with log.open('a', buffering=1) as stream:
    result = subprocess.run([
        sys.executable, '-u', '-m', 'tools.graspxl_matched_package',
        '--selection', str(root / 'out/grab-matched-graspxl/selected.json'),
        '--output', str(root.parent / 'GraspXL_SOMA_51Objects_GRABMatched'),
    ], cwd=root, stdout=stream, stderr=subprocess.STDOUT)
print(f'Build and validation exit code: {result.returncode}; log: {log}', flush=True)
sys.exit(result.returncode)
