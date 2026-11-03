"""Public split preparation and pure scoring helpers for English evaluation.

Task templates follow EleutherAI/lm-evaluation-harness: piqa/piqa.yaml,
hellaswag/utils.py, and lambada/lambada_openai.yaml. This native scorer is
not an official harness run. Generated data/reports must remain untracked.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TASKS = {
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1", "test", 4358),
    "lambada_openai": ("EleutherAI/lambada_openai", "default", "test", 5153),
    "piqa": ("baber/piqa", "default", "validation", 1838),
    "hellaswag": ("Rowan/hellaswag", "default", "validation", 10042),
}
CHOICE_TASKS = ("lambada_openai", "piqa", "hellaswag")
PROTOCOL = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def checkpoint_identity(directory):
    directory = Path(directory).expanduser().resolve()
    tracker = directory / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        step = tracker.read_text().strip()
        if not step.isdecimal():
            raise ValueError("Expected a numeric torch checkpoint iteration")
        directory /= f"iter_{int(step):07d}"
    if not directory.is_dir() or not re.fullmatch(r"iter_\d+", directory.name):
        raise ValueError(f"Expected a checkpoint root or iter_NNNNNNN directory: {directory}")
    shards = sorted(directory.glob("mp_rank_*/model_optim_rng.pt"))
    if not shards or any(p.stat().st_size == 0 for p in shards):
        raise ValueError(f"Missing or empty model checkpoint shards: {directory}")
    # The loader independently validates every TP/PP shard header and count.
    return dict(directory=str(directory), iteration=int(directory.name[5:]),
                files=[dict(name=str(p.relative_to(directory)), bytes=p.stat().st_size,
                            mtime_ns=p.stat().st_mtime_ns) for p in shards])


def parse_gpus(text):
    values = [value.strip() for value in text.split(",")]
    if not values or any(not x for x in values) or len(values) != len(set(values)):
        raise ValueError("--gpus must contain a nonempty list of distinct GPU IDs")
    if any(not (x.isdecimal() or x.startswith(("GPU-", "MIG-"))) for x in values):
        raise ValueError("GPU IDs must be numeric indices or GPU/MIG UUIDs")
    return values


def preprocess_hellaswag(text):
    return re.sub(r"\[.*?\]", "", text.strip().replace(" [title]", ". ")).replace("  ", " ")


def transform(task, doc):
    if task == "piqa":
        return "Question: " + doc["goal"] + "\nAnswer:", [doc["sol1"], doc["sol2"]], int(doc["label"])
    if task == "hellaswag":
        context = doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        return preprocess_hellaswag(context), [preprocess_hellaswag(s) for s in doc["endings"]], int(doc["label"])
    if task == "lambada_openai":
        context, target = doc["text"].rsplit(" ", 1)
        return context, [target], 0
    raise ValueError(f"Unknown choice task: {task}")


def encode_pair(tokenizer, context, choice, valid_vocab, max_length):
    continuation = " " + choice
    spaces = len(context) - len(context.rstrip())
    if spaces:
        continuation, context = context[-spaces:] + continuation, context[:-spaces]
    prefix = tokenizer.encode(context, add_special_tokens=False)
    whole = tokenizer.encode(context + continuation, add_special_tokens=False)
    if not prefix or len(whole) <= len(prefix) or whole[:len(prefix)] != prefix:
        raise ValueError("Empty target or inconsistent context/continuation token boundary")
    if any(token < 0 or token >= valid_vocab for token in whole):
        raise ValueError("Benchmark token outside the effective tokenizer vocabulary")
    if len(whole) - 1 > max_length:
        raise ValueError(f"Benchmark input has {len(whole)-1} tokens, exceeds context {max_length}; no truncation is allowed")
    return dict(input_ids=whole[:-1], targets=whole[len(prefix):],
                score_start=len(prefix)-1, chars=len(choice))


def fetch_page(spec, offset):
    import requests
    dataset, config, split, total = spec
    for attempt in range(4):
        try:
            response = requests.get("https://datasets-server.huggingface.co/rows",
                params=dict(dataset=dataset, config=config, split=split, offset=offset,
                            length=min(100, total-offset)), timeout=45)
            response.raise_for_status()
            result = response.json()
            rows = result["rows"]
            if result["num_rows_total"] != total or any(r["truncated_cells"] for r in rows):
                raise ValueError("Public split count changed or server truncated cell content")
            if [r["row_idx"] for r in rows] != list(range(offset, min(offset+100, total))):
                raise ValueError("Downloaded page contains missing/duplicate/unexpected row indices")
            return rows, result.get("dataset_git_revision")
        except (requests.RequestException, KeyError):
            if attempt == 3:
                raise
            time.sleep(min(2**attempt, 8))


def ensure_data(directory, offline=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sources = {}
    for task, spec in TASKS.items():
        path = directory / (task + ".jsonl")
        provenance = None
        if not path.exists():
            if offline:
                raise FileNotFoundError(f"Missing {path}; remove --offline to download the full public split")
            with ThreadPoolExecutor(max_workers=6) as executor:
                pages = list(executor.map(lambda offset: fetch_page(spec, offset), range(0, spec[3], 100)))
            revisions = {revision for _, revision in pages if revision}
            if len(revisions) > 1:
                raise ValueError("Dataset revision changed during download")
            docs = [r["row"] for rows, _ in pages for r in rows]
            tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
            tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in docs))
            tmp.replace(path)
            provenance = dict(dataset_revision=next(iter(revisions), None), downloaded_at=time.time())
        docs = read_jsonl(path)
        if len(docs) != spec[3]:
            raise ValueError(f"Expected complete {task} split with {spec[3]} rows, found {len(docs)}")
        for doc in docs:
            if task == "wikitext2":
                if not isinstance(doc.get("text"), str):
                    raise ValueError("Invalid WikiText row")
            else:
                _, choices, gold = transform(task, doc)
                expected = 2 if task == "piqa" else 4 if task == "hellaswag" else 1
                if len(choices) != expected or not 0 <= gold < len(choices) or any(not c for c in choices):
                    raise ValueError(f"Invalid choices or missing public gold label in {task}")
        sources[task] = dict(dataset=spec[0], config=spec[1], split=spec[2], records=len(docs),
                             path=str(path.resolve()), sha256=file_hash(path))
        if provenance:
            sources[task].update(provenance)
    return sources


def prepare_inputs(output, sources, tokenizer, context_length):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    valid_vocab = max(tokenizer.get_vocab().values()) + 1
    prepared = {}
    for task in CHOICE_TASKS:
        path = output / (task + ".jsonl")
        maximum = 0
        with path.open("w") as stream:
            for index, doc in enumerate(read_jsonl(sources[task]["path"])):
                context, choices, gold = transform(task, doc)
                requests = [encode_pair(tokenizer, context, c, valid_vocab, context_length) for c in choices]
                maximum = max(maximum, *(len(r["input_ids"]) for r in requests))
                stream.write(json.dumps(dict(index=index, gold=gold, requests=requests), separators=(",", ":")) + "\n")
        prepared[task] = dict(path=str(path.resolve()), sha256=file_hash(path), max_input_tokens=maximum)
    text = "\n\n".join(r["text"] for r in read_jsonl(sources["wikitext2"]["path"]))
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 2 or min(ids) < 0 or max(ids) >= valid_vocab:
        raise ValueError("Invalid WikiText token stream")
    path = output / "wikitext2-tokens.json"
    write_json(path, ids)
    prepared["wikitext2"] = dict(path=str(path.resolve()), sha256=file_hash(path), raw_tokens=len(ids))
    return prepared


def wiki_windows(ids, context_length, limit=0):
    stride = context_length - 1
    for index, start in enumerate(range(0, len(ids)-1, stride)):
        if limit and index >= limit:
            break
        yield index, ids[start:start+context_length]


def prediction(index, gold, requests, scores, task):
    if len(scores) != len(requests) or not scores:
        raise ValueError("Candidate score count mismatch")
    sums = [r["logprob_sum"] for r in scores]
    if not all(math.isfinite(x) for x in sums):
        raise ValueError("Non-finite benchmark scores")
    if [r["target_tokens"] for r in scores] != [len(r["targets"]) for r in requests]:
        raise ValueError("Candidate ordering or target token count mismatch")
    norm = [s/max(1, request["chars"]) for s, request in zip(sums, requests)]
    pred = max(range(len(sums)), key=sums.__getitem__)
    pred_norm = max(range(len(norm)), key=norm.__getitem__)
    record = dict(index=index, gold=gold, scores=scores, prediction=pred, prediction_norm=pred_norm,
                  correct=pred==gold, correct_norm=pred_norm==gold)
    if task == "lambada_openai":
        record.update(exact_match=scores[0]["greedy_match"], target_nll=-sums[0],
                      target_tokens=scores[0]["target_tokens"])
    return record


def aggregate(parts, counts, wiki_tokens, world_size):
    if len(parts) != world_size:
        raise ValueError("Missing worker results")
    stats = {k: sum(p["wikitext2"][k] for p in parts) for k in ("nll", "tokens", "correct")}
    if stats["tokens"] != wiki_tokens or wiki_tokens < 1:
        raise ValueError("WikiText coverage mismatch (padding or missing windows)")
    ce = stats["nll"]/wiki_tokens
    metrics = dict(wikitext2=dict(tokens=wiki_tokens, cross_entropy=ce, perplexity=math.exp(ce),
                                  top1_accuracy=stats["correct"]/wiki_tokens))
    records = {}
    for task in CHOICE_TASKS:
        expected = counts[task]
        for rank, part in enumerate(parts):
            actual = [r["index"] for r in part[task]]
            if actual != list(range(rank, expected, world_size)):
                raise ValueError(f"Wrong rank partition or missing/duplicate indices: {task}, rank{rank}")
        rows = sorted((r for p in parts for r in p[task]), key=lambda r:r["index"])
        if [r["index"] for r in rows] != list(range(expected)):
            raise ValueError(f"Incomplete split: {task}")
        records[task] = rows
        if task == "lambada_openai":
            nll = sum(r["target_nll"] for r in rows)
            tokens = sum(r["target_tokens"] for r in rows)
            correct = sum(r["exact_match"] for r in rows)
            metrics[task] = dict(samples=expected, acc=correct/expected, correct=correct,
                                 word_perplexity=math.exp(nll/expected), token_perplexity=math.exp(nll/tokens),
                                 target_tokens=tokens)
        else:
            correct, norm = sum(r["correct"] for r in rows), sum(r["correct_norm"] for r in rows)
            metrics[task] = dict(samples=expected, acc=correct/expected, acc_norm=norm/expected,
                                 correct=correct, correct_norm=norm)
    return metrics, records
