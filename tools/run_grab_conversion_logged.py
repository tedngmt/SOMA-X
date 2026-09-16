"""Run the resumable full-frame conversion with output redirected to disk."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
log = root / 'out' / 'grab-full-conversion.log'
log.parent.mkdir(exist_ok=True)
with log.open('a', buffering=1) as stream:
    result = subprocess.run([
        sys.executable, '-u', '-m', 'tools.convert_grab_to_soma',
        '--output', str(root.parent / 'GRAB_SOMA_51Objects'),
        '--batch-size', '128', '--resume',
    ], cwd=root, stdout=stream, stderr=subprocess.STDOUT)
print(f'Conversion exit code: {result.returncode}; log: {log}', flush=True)
sys.exit(result.returncode)
