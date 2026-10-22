"""Map sampled GPU peaks to each completed phase's log time window."""
import csv
import json
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
gpu = []
for row in csv.reader((HERE / 'gpu.csv').open()):
    try:
        gpu.append((datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp(),
                    int(row[1]), int(row[2])))
    except (ValueError, IndexError):
        continue
result = {}
for phase in ['baseline', 'reproduce', 'cp2-offload', 'cp1-control', 'cp2-checkpoint', 'cp2-resume']:
    log = HERE / f'{phase}.log'
    match = re.search(r'datetime: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)', log.read_text())
    if not match:
        continue
    start = datetime.fromisoformat(match[1]).timestamp()
    end = log.stat().st_mtime
    peaks = {str(i): max((v for t, rank, v in gpu if rank == i and start <= t <= end), default=None)
             for i in range(8)}
    result[phase] = dict(sampled_peak_memory_mib_by_rank=peaks, window_start=match[1],
                         window_end_unix=end, sample_interval_seconds=1)
(HERE / 'gpu-summary.json').write_text(json.dumps(result, indent=2) + '\n')
