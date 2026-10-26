"""Check text tokenization against the real teacher before allocating training GPUs."""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from rwkv_megatron.online_teacher import OnlineTeacherClient

PROBES = (
    "The quick brown fox jumps over the lazy dog. 12345",
    "พระบาทสมเด็จพระปรมินทรมหาจุฬาลงกรณ์ พระจุลจอมเกล้าเจ้าอยู่หัว",
    "हिन्दी भाषा में प्रशिक्षण और परीक्षण।",
    "العَرَبِيَّة لغة جميلة.",
    "Cafe\u0301 e\u0301lan: café déjà vu.",
    "中文和日本語のテスト。한국어 테스트. 🧪🌍",
)


def check_teacher_tokenizer(settings, prompts=PROBES):
    client = OnlineTeacherClient(
        server_url=settings.get("rwkv_teacher_url", "http://192.168.10.121:8000"),
        model_name=settings.get("rwkv_teacher_model", "Qwen3.8-27B"),
        tokenizer_path=settings.get("rwkv_teacher_tokenizer")
        or settings["tokenizer_model"],
        api_key=settings.get(
            "rwkv_teacher_api_key", os.environ.get("RWKV_TEACHER_API_KEY", "sk-distill")
        ),
        top_k=1,
        request_seed=settings.get("rwkv_teacher_seed", 1234),
        prompt_mode=settings.get("rwkv_teacher_prompt_mode", "text"),
        tokenizer_use_fast=settings.get("rwkv_teacher_tokenizer_use_fast", True),
        timeout=settings.get("rwkv_teacher_timeout", 600),
        max_retries=0,
        cache_size=0,
    )
    tokenizer = client._get_tokenizer()
    rows = []
    for index, prompt in enumerate(prompts):
        tokens = tokenizer(prompt, add_special_tokens=False).input_ids
        result = client.get_logits(torch.tensor([tokens]), prompts=[prompt])
        assert result.token_ids.shape[:2] == (1, len(tokens))
        rows.append(dict(probe=index, tokens=len(tokens), matched=True))
    return dict(
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_use_fast=client.tokenizer_use_fast,
        probes=rows,
        all_matched=True,
        optimizer_steps=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rwkv-config", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = yaml.safe_load(args.rwkv_config.read_text())
    prompts = [*PROBES, *(path.read_text() for path in args.prompt_file)]
    report = check_teacher_tokenizer(settings, prompts)
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
