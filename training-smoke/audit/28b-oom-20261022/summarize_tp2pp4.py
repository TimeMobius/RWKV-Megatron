"""Summarize TP2/PP4 short training and compare the prior TP4/PP2 run."""
import csv
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = "tp2pp4cp1-no-offload"
STEP = re.compile(
    r"iteration\s+(\d+)/\s*(\d+)\s*\|.*elapsed time per iteration \(ms\):\s*([\d.]+)"
    r".*lm loss:\s*([\d.Ee+-]+).*grad norm:\s*([\d.Ee+-]+)"
    r".*skipped iterations:\s*(\d+).*nan iterations:\s*(\d+)"
)


def steps_from_log(path):
    steps = []
    for line in path.read_text(errors="replace").splitlines():
        match = STEP.search(line)
        if match:
            step, total, ms, loss, grad, skip, nan = match.groups()
            steps.append(dict(step=int(step), total=int(total), ms=float(ms),
                              loss=float(loss), grad_norm=float(grad),
                              skipped=int(skip), nan=int(nan)))
    return steps


text = (HERE / f"{PHASE}.log").read_text(errors="replace")
steps = steps_from_log(HERE / f"{PHASE}.log")
rows = [json.loads(line) for path in sorted((HERE / PHASE).glob("rank*.jsonl"))
        for line in path.read_text().splitlines()]
finished = [row for row in rows if row["event"] == "after_step" and row["step"] == 30]
peaks = {str(i): 0 for i in range(8)}
with (HERE / f"{PHASE}-gpu.csv").open() as stream:
    for row in csv.reader(stream):
        try:
            rank, memory = str(int(row[1])), int(row[2])
        except (ValueError, IndexError):
            continue
        peaks[rank] = max(peaks[rank], memory)
exit_info = json.loads((HERE / f"{PHASE}-exit.json").read_text())
evaluations = [dict(step=int(step), loss=float(loss), perplexity=float(ppl))
               for step, loss, ppl in re.findall(
                   r"validation loss at iteration (\d+)[^\n]*lm loss value:\s*([\d.Ee+-]+)"
                   r"\s*\|\s*lm loss PPL:\s*([\d.Ee+-]+)", text)]
finite = bool(steps) and all(math.isfinite(row[key]) for row in steps
                             for key in ["loss", "grad_norm"])
zero_skips = all(row["nan"] == row["skipped"] == 0 for row in steps)
passed = (exit_info["exit_code"] == 0 and len(steps) == 30
          and len(finished) == 8 and bool(evaluations) and finite and zero_skips)
reference = {row["step"]: row for row in steps_from_log(HERE / "tp4pp2cp1-no-offload.log")}
pairs = [dict(step=row["step"], loss_abs_difference=abs(row["loss"] - reference[row["step"]]["loss"]),
              grad_norm_relative_difference=abs(row["grad_norm"] - reference[row["step"]]["grad_norm"])
              / max(abs(reference[row["step"]]["grad_norm"]), 1e-30))
         for row in steps if row["step"] in reference]
result = dict(
    status="passed" if passed else "failed", **exit_info,
    tp=2, pp=4, cp=1, sequence_parallel=True, seq_length=4096,
    micro_batch_size=1, global_batch_size=2,
    optimizer_cpu_offload=False, optimizer_offload_fraction=0.0,
    use_precision_aware_optimizer=False, train_steps=len(steps),
    completed_ranks_at_step30=sorted(row["rank"] for row in finished),
    logged_loss_and_grad_finite=finite, zero_nan_and_skips=zero_skips,
    all_recorded_storage_cuda_at_step30=(len(finished) == 8 and all(
        "/cuda/" in key for row in finished for key in row["storage_bytes"])),
    median_step_ms_after_step2=statistics.median([row["ms"] for row in steps if row["step"] > 2]) if len(steps) > 2 else None,
    max_cuda_allocated_gib=max((row["max_allocated"] / 2**30 for row in rows), default=None),
    sampled_gpu_peak_mib=max(peaks.values()), sampled_gpu_peak_mib_by_rank=peaks,
    static_storage_gib_at_step30_by_rank={str(row["rank"]): sum(row["storage_bytes"].values()) / 2**30 for row in finished},
    final_train_loss=steps[-1]["loss"] if steps else None,
    evaluations=evaluations,
    comparison_to_tp4pp2=dict(
        max_logged_loss_abs_difference=max((p["loss_abs_difference"] for p in pairs), default=None),
        max_logged_grad_norm_relative_difference=max((p["grad_norm_relative_difference"] for p in pairs), default=None),
        note="Logged scalars only; changes TP and PP; not full-tensor numerical equivalence."),
    steps=steps,
    audited_oom_ranks=sorted(row["rank"] for row in rows if row["event"] == "oom"),
)
(HERE / f"{PHASE}-results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({key: value for key, value in result.items() if key != "steps"}, indent=2))
