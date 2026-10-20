"""Load RWKV YAML settings on top of upstream Megatron argument defaults."""

import argparse
import math

import yaml


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('L2Wrap coefficient must be finite and nonnegative')
    return value


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('value must be a positive integer')
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('value must be a nonnegative integer')
    return value


def add_rwkv_arguments(parser):
    parser.add_argument("--rwkv-config", help="RWKV YAML configuration")
    parser.add_argument("--load-pretrained-rwkv")
    parser.add_argument("--load-pretrained-rwkv-source-type", default="rwkv-lm")
    parser.add_argument("--check-against-pretrained", action="store_true")
    parser.add_argument("--auto-train-iters", action="store_true")
    parser.add_argument("--debug-dump", action="store_true")
    parser.add_argument("--shuffle-blend-datasets", action="store_true")
    parser.add_argument("--rwkv-l2wrap-coefficient", type=nonnegative_float, default=0.0,
                        help="RWKV max-logit gradient strength; 0 disables, official value is 1e-4")
    parser.add_argument("--rwkv-key-fusion", action=argparse.BooleanOptionalAction, default=None,
                        help="Fuse FP32 key calculation; overrides RWKV7_KEY_FUSION when set")
    parser.add_argument("--fixed-eval-samples", action=argparse.BooleanOptionalAction, default=False,
                        help="Restart validation at the same first samples on every evaluation")
    parser.add_argument("--rwkv-fp8-scope", choices=("ffn", "all"), default="ffn",
                        help="FP8 FFN only (default), or all six large linears (experimental)")
    parser.add_argument("--rwkv-distill", action=argparse.BooleanOptionalAction, default=False,
                        help="Align RWKV logits with an online Qwen teacher")
    parser.add_argument("--rwkv-teacher-url", default="http://192.168.10.121:8000")
    parser.add_argument("--rwkv-teacher-model", default="Qwen3.8-27B")
    parser.add_argument("--rwkv-teacher-tokenizer", default=None)
    parser.add_argument("--rwkv-teacher-top-k", type=int, default=100)
    parser.add_argument("--rwkv-teacher-timeout", type=float, default=600.0)
    parser.add_argument("--rwkv-teacher-max-retries", type=nonnegative_int, default=3)
    parser.add_argument("--rwkv-teacher-retry-backoff-initial", type=float, default=1.0)
    parser.add_argument("--rwkv-teacher-retry-backoff-max", type=float, default=30.0)
    parser.add_argument("--rwkv-teacher-retry-jitter", type=float, default=0.2)
    parser.add_argument("--rwkv-teacher-cache-size", type=int, default=256)
    parser.add_argument("--rwkv-teacher-async-prefetch", action=argparse.BooleanOptionalAction,
                        default=False, help="Overlap concurrent teacher requests with student training")
    parser.add_argument("--rwkv-teacher-prefetch-workers", type=positive_int, default=16)
    parser.add_argument("--rwkv-teacher-prefetch-lookahead", type=positive_int, default=32)
    parser.add_argument("--rwkv-teacher-prefetch-min-workers", type=positive_int, default=1)
    parser.add_argument("--rwkv-teacher-prefetch-latency-target", type=float, default=2.0)
    parser.add_argument("--rwkv-distill-temperature", type=float, default=2.0)
    parser.add_argument("--rwkv-distill-valid-vocab-size", type=positive_int, default=None,
                        help="Effective teacher vocabulary size; defaults to the tokenizer ID range")
    parser.add_argument("--rwkv-distill-logit-weight", type=float, default=1.0)
    parser.add_argument("--rwkv-distill-ce-weight", type=float, default=0.0)
    parser.add_argument("--rwkv-distill-filter-roundtrip", action=argparse.BooleanOptionalAction,
                        default=False, help="Preselect samples whose decoded text preserves teacher token IDs")
    parser.add_argument("--rwkv-jsonl-text-field", default="text")
    parser.add_argument("--rwkv-jsonl-max-records", type=nonnegative_int, default=0)
    parser.add_argument("--rwkv-jsonl-chunk-chars", type=positive_int, default=512)
    for name in ("D_AAA_LORA", "D_DECAY_LORA", "D_GATE_LORA", "D_MV_LORA"):
        parser.add_argument(f"--{name}", type=int)

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--rwkv-config")
    known, _ = probe.parse_known_args()
    if known.rwkv_config is None:
        return parser
    with open(known.rwkv_config) as source:
        config = yaml.safe_load(source)
    language = config.get("language_model", {})
    parallel = config.get("model_parallel", {})
    flat = {key: value for key, value in config.items() if key not in ("language_model", "model_parallel")}
    flat.update(parallel)
    flat.update(language)
    defaults = vars(parser.parse_args([]))
    # Old YAMLs contain optional internal fields set to null. Retain current
    # upstream defaults for those fields, instead of nulling new requirements.
    settings = {key: value for key, value in flat.items() if key in defaults and (value is not None or defaults[key] is None)}
    settings.update(
        num_experts=language.get("num_moe_experts"),
        no_persist_layer_norm=not language.get("persist_layer_norm", True),
        no_initialization=not parallel.get("perform_initialization", True),
        decoder_first_pipeline_num_layers=language.get("num_layers_in_first_pipeline_stage"),
        decoder_last_pipeline_num_layers=language.get("num_layers_in_last_pipeline_stage"),
        swiglu=language.get("activation_func") == "swiglu",
        squared_relu=language.get("activation_func") == "squaredrelu",
        bias_swiglu_fusion=parallel.get("bias_swiglu_fusion", False),
    )
    if not language.get("fp8"):
        settings["fp8"] = None
    elif language["fp8"] not in ("hybrid", "e4m3"):
        raise ValueError("language_model.fp8 must be 'hybrid', 'e4m3', or false")
    if 'rwkv_l2wrap_coefficient' in settings:
        settings['rwkv_l2wrap_coefficient'] = nonnegative_float(settings['rwkv_l2wrap_coefficient'])
    if settings.get('rwkv_key_fusion') is not None and not isinstance(settings['rwkv_key_fusion'], bool):
        raise ValueError('rwkv_key_fusion must be a YAML boolean (true/false)')
    if not isinstance(settings.get('fixed_eval_samples', False), bool):
        raise ValueError('fixed_eval_samples must be a YAML boolean (true/false)')
    if settings.get('rwkv_fp8_scope', 'ffn') not in ('ffn', 'all'):
        raise ValueError("rwkv_fp8_scope must be 'ffn' or 'all'")
    if settings.get('rwkv_teacher_top_k', 100) < 1 or settings.get('rwkv_teacher_top_k', 100) > 1000:
        raise ValueError("rwkv_teacher_top_k must be in [1, 1000]")
    for key in ('rwkv_teacher_retry_backoff_initial', 'rwkv_teacher_retry_backoff_max',
                'rwkv_teacher_retry_jitter'):
        if settings.get(key, 0.0) < 0:
            raise ValueError(f'{key} must be nonnegative')
    if settings.get('rwkv_teacher_retry_backoff_max', 30.0) < settings.get('rwkv_teacher_retry_backoff_initial', 1.0):
        raise ValueError('rwkv_teacher_retry_backoff_max must be >= initial backoff')
    if settings.get('rwkv_teacher_cache_size', 256) < 0:
        raise ValueError('rwkv_teacher_cache_size must be nonnegative')
    if not isinstance(settings.get('rwkv_teacher_async_prefetch', False), bool):
        raise ValueError('rwkv_teacher_async_prefetch must be a YAML boolean (true/false)')
    for key in ('rwkv_teacher_prefetch_workers', 'rwkv_teacher_prefetch_lookahead'):
        value = settings.get(key, 16 if key.endswith('workers') else 32)
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if settings.get('rwkv_teacher_prefetch_min_workers', 1) > settings.get('rwkv_teacher_prefetch_workers', 16):
        raise ValueError('rwkv_teacher_prefetch_min_workers must be <= rwkv_teacher_prefetch_workers')
    if settings.get('rwkv_teacher_prefetch_latency_target', 2.0) <= 0:
        raise ValueError('rwkv_teacher_prefetch_latency_target must be positive')
    if settings.get('rwkv_jsonl_max_records', 0) < 0:
        raise ValueError('rwkv_jsonl_max_records must be nonnegative')
    if settings.get('rwkv_jsonl_chunk_chars', 512) < 1:
        raise ValueError('rwkv_jsonl_chunk_chars must be positive')
    for key in ('rwkv_distill_temperature', 'rwkv_distill_logit_weight', 'rwkv_distill_ce_weight'):
        if settings.get(key, 0.0) < 0:
            raise ValueError(f"{key} must be nonnegative")
    valid_vocab_size = settings.get('rwkv_distill_valid_vocab_size')
    if valid_vocab_size is not None and (type(valid_vocab_size) is not int or valid_vocab_size < 1):
        raise ValueError('rwkv_distill_valid_vocab_size must be a positive integer or null')
    # Position encoding is selected by position_embedding_type upstream.
    settings["add_position_embedding"] = True
    parser.set_defaults(**settings)
    return parser
