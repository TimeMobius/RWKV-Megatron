#!/usr/bin/env python
"""One-command native RWKV English benchmarks, run in the training environment.

Examples (default mode is a 32-example integration smoke, not full scores):
  python tools/evaluate_english_benchmarks.py --rwkv-config CONFIG --gpus 0,1,2,3
  python tools/evaluate_english_benchmarks.py --rwkv-config CONFIG --mode full
  python tools/evaluate_english_benchmarks.py --rwkv-config CONFIG --output RUN --resume

Checkpoint defaults to CONFIG.save/CONFIG.load; complete cached public splits
are validated or downloaded automatically. Pass --offline to forbid downloads.
Each GPU holds a complete inference replica; this is sample-parallel evaluation,
not TP/PP inference. No teacher, optimizer, or training updates are involved.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.english_benchmark_data import (
    CHOICE_TASKS, PROTOCOL, TASKS, aggregate, checkpoint_identity, digest,
    ensure_data, file_hash, parse_gpus, prediction, prepare_inputs, read_jsonl,
    wiki_windows, write_json,
)


class Scorer:
    def __init__(self, model, valid_vocab, context_length=512, batch_size=4, device="cuda"):
        import torch
        self.model, self.valid_vocab = model, valid_vocab
        self.context_length, self.batch_size = context_length, batch_size
        self.device = torch.device(device)

    def _forward(self, requests):
        import torch
        lengths = [len(r["input_ids"]) for r in requests]
        if not lengths or min(lengths) < 1 or max(lengths) > self.context_length:
            raise ValueError("Empty input or context overflow; no truncation is allowed")
        width = ((max(lengths)+63)//64)*64
        tokens = torch.zeros((len(requests), width), dtype=torch.long, device=self.device)
        for row, request in enumerate(requests):
            tokens[row, :lengths[row]] = torch.tensor(request["input_ids"], device=self.device)
        positions = torch.arange(width, device=self.device)[None].expand_as(tokens)
        logits = self.model(tokens, positions, None)
        if logits.ndim != 3 or logits.shape[:2] != tokens.shape or logits.shape[-1] < self.valid_vocab:
            raise ValueError("Expected model logits [batch, sequence, vocab]")
        return logits

    def score_requests(self, requests):
        import torch
        ordered = sorted(enumerate(requests), key=lambda x:len(x[1]["input_ids"]))
        scores = [None]*len(requests)
        with torch.inference_mode():
            for start in range(0, len(ordered), self.batch_size):
                batch = ordered[start:start+self.batch_size]
                logits = self._forward([r for _, r in batch])
                for row, (index, request) in enumerate(batch):
                    count, begin = len(request["targets"]), request["score_start"]
                    if count < 1 or begin < 0 or begin+count != len(request["input_ids"]):
                        raise ValueError("Scoring span does not align with actual input/targets")
                    selected = logits[row, begin:begin+count, :self.valid_vocab].float()
                    targets = torch.tensor(request["targets"], dtype=torch.long, device=self.device)
                    logp = selected.gather(-1, targets[:, None]).squeeze(-1)-torch.logsumexp(selected, -1)
                    if not torch.isfinite(logp).all():
                        raise ValueError("Non-finite continuation likelihood")
                    scores[index] = dict(logprob_sum=float(logp.double().sum()), target_tokens=count,
                                         greedy_match=bool(torch.equal(selected.argmax(-1), targets)))
                del logits
        return scores

    def score_wiki(self, ids):
        import torch
        import torch.nn.functional as F
        with torch.inference_mode():
            logits = self._forward([dict(input_ids=ids[:-1])])[0]
            labels = torch.tensor(ids[1:], dtype=torch.long, device=self.device)
            nll, correct = 0.0, 0
            for start in range(0, len(labels), 64):
                targets = labels[start:start+64]
                values = logits[start:start+len(targets), :self.valid_vocab].float()
                losses = F.cross_entropy(values, targets, reduction="none")
                if not torch.isfinite(losses).all():
                    raise ValueError("Non-finite WikiText NLL")
                nll += float(losses.double().sum())
                correct += int((values.argmax(-1)==targets).sum())
        return dict(nll=nll, correct=correct, tokens=len(ids)-1)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rwkv-config", type=Path)
    p.add_argument("--checkpoint", type=Path, help="Torch checkpoint root or explicit iter_NNNNNNN")
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    p.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    p.add_argument("--limit", type=int, help="Smoke examples/windows per task (default32); forbidden in full mode")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--save-every", type=int, default=100)
    cache = ROOT/"training-smoke/english-benchmark-cache"
    p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("RWKV_BENCHMARK_DATA_DIR", cache)))
    p.add_argument("--output", type=Path)
    p.add_argument("--offline", action="store_true", help="Require existing full JSONL data and local tokenizer")
    p.add_argument("--resume", action="store_true", help="Resume only if checkpoint, data, tokenizer and settings match")
    p.add_argument("--prepare-only", action="store_true", help="Validate/cache full splits without loading GPU weights")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def validate_options(args):
    if args.context_length < 64 or args.context_length % 64:
        raise ValueError("--context-length must be a positive multiple of64")
    if args.batch_size < 1 or args.save_every < 1:
        raise ValueError("--batch-size and --save-every must be positive")
    if args.mode == "full" and args.limit is not None:
        raise ValueError("--limit is forbidden in full mode; use --mode smoke")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive in smoke mode")
    return 0 if args.mode == "full" else args.limit or 32


def prepare(args, cfg):
    from transformers import AutoTokenizer
    limit = validate_options(args)
    gpus = parse_gpus(args.gpus)
    checkpoint = args.checkpoint or cfg.get("save") or cfg.get("load")
    if not checkpoint:
        raise ValueError("Pass --checkpoint or configure save/load")
    ckpt = checkpoint_identity(checkpoint)
    tokenizer_path = args.tokenizer or cfg.get("rwkv_teacher_tokenizer") or cfg.get("tokenizer_model")
    if not tokenizer_path:
        raise ValueError("Pass --tokenizer or configure tokenizer_model")
    output = (args.output or ROOT/"training-smoke"/("english-benchmarks-"+datetime.now().strftime("%Y%m%d-%H%M%S"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output/"manifest.json"
    if manifest_file.exists() and not args.resume:
        raise ValueError(f"{output} already has a manifest; use --resume or a new --output")
    if args.resume and not manifest_file.exists():
        raise ValueError("--resume requires an existing manifest")
    sources = ensure_data(args.data_dir.expanduser().resolve(), args.offline)
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=args.offline,
                                        trust_remote_code=True, use_fast=False)
    identity = dict(protocol=PROTOCOL, checkpoint=ckpt, mode=args.mode, sample_limit=limit,
                    world_size=len(gpus), context_length=args.context_length, batch_size=args.batch_size,
                    tokenizer=str(tokenizer_path), tokenizer_class=type(tok).__name__,
                    vocabulary_sha256=digest(tok.get_vocab()), valid_vocab=max(tok.get_vocab().values())+1,
                    scorer_sha256=file_hash(Path(__file__).resolve()),
                    preparation_sha256=file_hash(ROOT/"tools/english_benchmark_data.py"),
                    sources={k:{x:d[x] for x in ("dataset", "config", "split", "records", "sha256")} for k,d in sources.items()})
    fingerprint = digest(identity)
    if args.resume:
        manifest = json.loads(manifest_file.read_text())
        if manifest.get("fingerprint") != fingerprint or manifest["identity"] != identity:
            raise ValueError("Cannot resume: checkpoint/data/tokenizer/parallelism/scoring settings changed")
        for data in manifest["prepared"].values():
            if file_hash(data["path"]) != data["sha256"]:
                raise ValueError("Prepared benchmark data changed")
    else:
        prepared = prepare_inputs(output/"prepared", sources, tok, args.context_length)
        manifest = dict(identity=identity, fingerprint=fingerprint, prepared=prepared,
                         source_files=sources, save_every=args.save_every,
                         scoring=dict(zero_shot=True, sampling=False, truncate=False,
                           templates="EleutherAI/lm-evaluation-harness: piqa, hellaswag, lambada_openai",
                           acc_norm="Sum continuation loglikelihood divided by original choice character count",
                           lambada="All final-word tokens must match greedy argmax, including punctuation/case; word PPL",
                           wikitext="Raw test text joined with two newlines; reset context; stride=context-1; every token except first scored once",
                           runtime="Custom native Megatron scorer, not an official harness run"))
        write_json(manifest_file, manifest)
    return output, gpus, manifest


def worker(output):
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from tools.evaluate_distill_checkpoint import checkpoint_native, build_model

    manifest = json.loads((output/"manifest.json").read_text())
    settings = manifest["identity"]
    rank, world, local = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world != settings["world_size"]:
        raise ValueError("torchrun world size differs from evaluation manifest")
    if checkpoint_identity(settings["checkpoint"]["directory"]) != settings["checkpoint"]:
        raise ValueError("Checkpoint changed after preparation")
    torch.set_num_threads(4)
    torch._dynamo.config.cache_size_limit = 256
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl")
    try:
        ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(1234)
        native, metadata, step, folder = checkpoint_native(settings["checkpoint"]["directory"])
        if settings["valid_vocab"] > metadata.padded_vocab_size or step != settings["checkpoint"]["iteration"]:
            raise ValueError("Checkpoint iteration or vocabulary mismatch")
        model = build_model(metadata)
        model.from_rwkv_lm(native, False)
        del native
        gc.collect()
        scorer = Scorer(model, settings["valid_vocab"], settings["context_length"], settings["batch_size"], f"cuda:{local}")
        partial = output/"partials"/f"rank-{rank:03d}.json"
        state = json.loads(partial.read_text()) if partial.exists() else dict(fingerprint=manifest["fingerprint"], rank=rank, world_size=world, tasks={})
        if state.get("fingerprint") != manifest["fingerprint"] or state.get("rank") != rank or state.get("world_size") != world:
            raise ValueError("Partial result identity mismatch")
        ids = json.loads(Path(manifest["prepared"]["wikitext2"]["path"]).read_text())
        windows = list(wiki_windows(ids, settings["context_length"], settings["sample_limit"]))
        own = [(i,w) for i,w in windows if i%world==rank]
        saved = state.setdefault("wiki_windows", [])
        if [r["index"] for r in saved] != [i for i,_ in own[:len(saved)]]:
            raise ValueError("Corrupt or incompatible WikiText resume indices")
        for position, (index, window) in enumerate(own[len(saved):], start=len(saved)+1):
            saved.append(dict(index=index, **scorer.score_wiki(window)))
            if position%manifest["save_every"] == 0:
                write_json(partial, state)
        state["wikitext2"] = {k:sum(r[k] for r in saved) for k in ("nll", "tokens", "correct")}
        write_json(partial, state)
        for task in CHOICE_TASKS:
            rows = read_jsonl(manifest["prepared"][task]["path"])
            if settings["sample_limit"]:
                rows = rows[:settings["sample_limit"]]
            own = [r for r in rows if r["index"]%world==rank]
            records = state["tasks"].setdefault(task, [])
            if [r["index"] for r in records] != [r["index"] for r in own[:len(records)]]:
                raise ValueError(f"Corrupt resume indices for {task}")
            for position, row in enumerate(own[len(records):], start=len(records)+1):
                scores = scorer.score_requests(row["requests"])
                records.append(prediction(row["index"], row["gold"], row["requests"], scores, task))
                if position%manifest["save_every"] == 0:
                    write_json(partial, state)
                    print(f"rank={rank} {task} {position}/{len(own)}", flush=True)
            write_json(partial, state)
            print(f"rank={rank} {task} complete {len(records)} rows", flush=True)
        state.update(completed=True, peak_memory_bytes=torch.cuda.max_memory_allocated())
        write_json(partial, state)
        dist.barrier(device_ids=[local])
        if rank == 0:
            parts = [json.loads((output/"partials"/f"rank-{i:03d}.json").read_text()) for i in range(world)]
            if any(p.get("fingerprint") != manifest["fingerprint"] or not p.get("completed") for p in parts):
                raise ValueError("Worker result missing or belongs to a different run")
            for i, part in enumerate(parts):
                expected_indices = [index for index,_ in windows if index%world==i]
                if part["rank"] != i or [w["index"] for w in part["wiki_windows"]] != expected_indices:
                    raise ValueError("WikiText worker rank/window coverage mismatch")
            counts = {task:min(TASKS[task][3], settings["sample_limit"]) if settings["sample_limit"] else TASKS[task][3] for task in CHOICE_TASKS}
            wiki_tokens = sum(len(w)-1 for _,w in windows)
            metrics, records = aggregate([{**p, **p["tasks"]} for p in parts], counts, wiki_tokens, world)
            write_json(output/"results.json", dict(status="completed", fingerprint=manifest["fingerprint"],
                iteration=step, checkpoint=str(folder), mode=settings["mode"], full_splits=settings["sample_limit"]==0,
                world_size=world, metrics=metrics, records=records,
                peak_memory_bytes=[p["peak_memory_bytes"] for p in parts],
                coverage=dict(all_ranks_verified=True, unique_complete_indices=True, no_truncation=True,
                              wiki_windows=len(windows), wiki_target_tokens=wiki_tokens)))
        dist.barrier(device_ids=[local])
    finally:
        dist.destroy_process_group()


def launch(output, gpus):
    from tools.launch_distillation import stop_process_group
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={len(gpus)}",
               str(Path(__file__).resolve()), "--worker", "--output", str(output)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
    process = None
    handlers = {}
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Evaluation interrupted by signal{signum}")
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupt)
        with (output/"evaluation.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait():
                raise RuntimeError(f"Evaluation failed ({process.returncode}); see {output/'evaluation.log'}; rerun with --resume")
    finally:
        stop_process_group(process)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def report_and_upload(output, cfg, mode):
    report = json.loads((output/"results.json").read_text())
    lines = [f"# English benchmarks: step{report['iteration']} ({report['mode']})", "",
             "Full public splits." if report["full_splits"] else "Integration smoke only: these are NOT full benchmark scores.",
             "", "| Task | Scored | Metrics |", "|---|---:|---|"]
    for task, m in report["metrics"].items():
        values = "; ".join(f"{k}={v:.6g}" for k,v in m.items() if k not in ("samples", "tokens"))
        lines.append(f"| {task} | {m.get('samples', m.get('tokens'))} | {values} |")
    lines += ["", "Sample-parallel GPU replicas; native Megatron BF16 inference. No teacher or optimizer updates.",
              "Task templates follow lm-evaluation-harness; this is not an official harness run.",
              "WikiText: Qwen-tokenizer-dependent raw test PPL, reset context, stride=context-1; excludes padding.",
              "LAMBADA PPL is per final word; token PPL is supplementary. acc_norm uses choice character count.",
              "No training-data contamination audit has been performed.", "", f"Checkpoint: {report['checkpoint']}"]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n")
    if mode == "disabled":
        write_json(output/"wandb.json", dict(mode=mode, uploaded=False))
        return
    import wandb
    if not cfg.get("wandb_project"):
        raise ValueError("W&B reporting requires wandb_project in the config")
    runid = digest(dict(output=str(output), fingerprint=report["fingerprint"]))[:16]
    with wandb.init(entity=cfg.get("wandb_entity"), project=cfg["wandb_project"], id=runid,
                    resume="allow", mode=mode, name=f"english-{report['mode']}-step{report['iteration']}",
                    job_type="evaluation" if report["full_splits"] else "evaluation-smoke", dir=str(output),
                    config=dict(iteration=report["iteration"], full_splits=report["full_splits"],
                                world_size=report["world_size"], fingerprint=report["fingerprint"], teacher_requests=0, optimizer_steps=0)) as run:
        for task, metrics in report["metrics"].items():
            for key, value in metrics.items():
                run.summary[f"english/{task}/{key}"] = value
        run.summary["english/full_splits"] = report["full_splits"]
        run.summary["english/coverage_passed"] = True
        artifact = wandb.Artifact("rwkv-english-benchmarks-"+report["mode"], type="evaluation")
        for name in ("results.json", "manifest.json", "REPORT.md"):
            artifact.add_file(str(output/name), name=name)
        run.log_artifact(artifact)
        url = run.url
    write_json(output/"wandb.json", dict(id=runid, mode=mode, uploaded=mode=="online", url=url))


def main():
    args = parser().parse_args()
    if args.worker:
        if args.output is None:
            raise ValueError("Worker requires --output")
        worker(args.output.resolve())
        return
    if args.rwkv_config is None:
        parser().error("--rwkv-config is required")
    import yaml
    cfg = yaml.safe_load(args.rwkv_config.read_text())
    output, gpus, manifest = prepare(args, cfg)
    print(f"Prepared {args.mode} evaluation on {len(gpus)} GPUs: {output}", flush=True)
    if args.prepare_only:
        return
    result = output/"results.json"
    if not result.exists():
        launch(output, gpus)
    report = json.loads(result.read_text())
    if report.get("status") != "completed" or report.get("fingerprint") != manifest["fingerprint"]:
        raise ValueError("Completed result identity mismatch")
    mode = args.wandb_mode or os.environ.get("WANDB_MODE") or ("online" if cfg.get("wandb_project") else "disabled")
    report_and_upload(output, cfg, mode)
    print(f"Completed {args.mode} evaluation: {output/'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
