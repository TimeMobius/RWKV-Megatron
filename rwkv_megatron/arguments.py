"""Load RWKV YAML settings on top of upstream Megatron argument defaults."""

import argparse
import math

import yaml


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('L2Wrap coefficient must be finite and nonnegative')
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
    if 'rwkv_l2wrap_coefficient' in settings:
        settings['rwkv_l2wrap_coefficient'] = nonnegative_float(settings['rwkv_l2wrap_coefficient'])
    if settings.get('rwkv_key_fusion') is not None and not isinstance(settings['rwkv_key_fusion'], bool):
        raise ValueError('rwkv_key_fusion must be a YAML boolean (true/false)')
    # Position encoding is selected by position_embedding_type upstream.
    settings["add_position_embedding"] = True
    parser.set_defaults(**settings)
    return parser
