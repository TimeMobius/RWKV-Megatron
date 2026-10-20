#!/usr/bin/env python3
"""Retokenize a bounded MiniPile subset with the Qwen tokenizer.

The checked-in MiniPile index is encoded with the old Xiaoke tokenizer.  A
Qwen-vocabulary student must use this conversion before training; feeding the
old IDs directly would silently corrupt the teacher/student alignment.
"""

import argparse
import os

import torch
from transformers import AutoTokenizer

from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="/mnt/data/Datasets/minipile/minipile_text_document")
    p.add_argument("--output", default="/mnt/data/Datasets/minipile/minipile_qwen_text_document")
    p.add_argument("--source-tokenizer", default="/hikscale/models/XiaoKe/Xiaoke-5-13B-2608")
    p.add_argument("--target-tokenizer", default="/hikscale/models/Qwen/Qwen3.8-27B")
    p.add_argument("--max-documents", type=int, default=10000)
    args = p.parse_args()

    source_tok = AutoTokenizer.from_pretrained(args.source_tokenizer, trust_remote_code=True)
    target_tok = AutoTokenizer.from_pretrained(args.target_tokenizer, trust_remote_code=True)
    source = IndexedDataset(args.source, mmap=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    builder = IndexedDatasetBuilder(args.output + ".bin", dtype=__import__("numpy").int32)
    count = min(len(source), args.max_documents)
    total = 0
    for i in range(count):
        old_ids = source.get(i).tolist()
        text = source_tok.decode(old_ids, skip_special_tokens=False)
        new_ids = target_tok(text, add_special_tokens=False).input_ids
        if not new_ids:
            new_ids = [target_tok.eos_token_id]
        builder.add_item(torch.tensor(new_ids, dtype=torch.int32))
        builder.end_document()
        total += len(new_ids)
        if (i + 1) % 1000 == 0:
            print(f"processed {i + 1}/{count} documents, {total} tokens", flush=True)
    builder.finalize(args.output + ".idx")
    print(f"wrote {count} documents and {total} Qwen tokens to {args.output}")


if __name__ == "__main__":
    main()
