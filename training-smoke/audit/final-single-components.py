"""Replay frozen kernel checks into a new directory, preserving earlier results."""

import json
import runpy
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
OUTPUT = AUDIT / 'final-single-20261012'


def main():
    OUTPUT.mkdir(exist_ok=True)
    for script in ('kernel.py', 'blinkdl-kernel.py'):
        namespace = runpy.run_path(str(AUDIT / script))
        namespace['main'].__globals__['PATH'] = OUTPUT
        namespace['main']()
    for name in ('kernel-results.json', 'blinkdl-kernel-results.json'):
        rows = json.loads((OUTPUT / name).read_text())
        assert all(row['passed'] for row in rows), name
    print('FINAL SINGLE-GPU KERNEL CHECKS PASSED')


if __name__ == '__main__':
    main()
