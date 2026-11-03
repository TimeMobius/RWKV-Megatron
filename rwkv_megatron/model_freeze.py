"""RWKV parameter freezing, applied before DDP and optimizer construction.

Extra patterns use fnmatch against Megatron names and native RWKV aliases.
Precedence: component switches < freeze patterns < train patterns. All switches
are opt-in, so existing configurations keep their previous training behavior.
"""

from contextlib import contextmanager
from fnmatch import fnmatchcase
from types import SimpleNamespace
from unittest.mock import patch

FREEZE_COMPONENTS = (
    'embedding', 'lm_head', 'attn_norm', 'ffn_norm', 'final_norm',
    'ffn_value', 'ffn_key', 'ffn_receptance', 'attn_receptance',
    'attn_key', 'attn_value', 'attn_output', 'token_shift', 'lora',
)


def freeze_policy(args):
    """Stable checkpoint signature; missing fields in older runs mean disabled."""
    result = {name: bool(getattr(args, f'rwkv_freeze_{name}', False))
              for name in FREEZE_COMPONENTS}
    for kind in ('freeze', 'train'):
        result[f'{kind}_extra_patterns'] = sorted(set(
            getattr(args, f'rwkv_{kind}_extra_patterns', None) or []))
    return result


def validate_freeze_settings(settings):
    for name in FREEZE_COMPONENTS:
        key = f'rwkv_freeze_{name}'
        if type(settings.get(key, False)) is not bool:
            raise ValueError(f'{key} must be a YAML boolean (true/false)')
    for kind in ('freeze', 'train'):
        key = f'rwkv_{kind}_extra_patterns'
        value = settings.get(key, [])
        if not isinstance(value, list) or any(not isinstance(p, str) or not p for p in value):
            raise ValueError(f'{key} must be a list of nonempty fnmatch patterns')


def _component(name):
    if name.startswith('embedding.word_embeddings.'):
        return 'embedding'
    if name.startswith('output_layer.'):
        return 'lm_head'
    if '.input_layernorm.' in name:
        return 'attn_norm'
    if '.pre_mlp_layernorm.' in name:
        return 'ffn_norm'
    if name.startswith('decoder.final_layernorm.'):
        return 'final_norm'
    if '.mlp.linear_up.' in name:
        return 'ffn_key'  # Native FFN key expands hidden -> FFN, despite the name 'up'.
    if '.mlp.linear_down.' in name:
        return 'ffn_value'
    for projection in ('receptance', 'key', 'value'):
        if f'.self_attention.linear_{projection}.' in name:
            return f'attn_{projection}'
    if '.self_attention.linear_proj.' in name:
        return 'attn_output'
    if name.endswith('.self_attention.mix_coeff_input'):
        return 'token_shift'
    if '.self_attention.bottleneck_' in name:
        return 'lora'  # Includes w0/a0/v0 bias, as in the original config.
    return None


def parameter_aliases(model, name):
    """Use global layer IDs for PP; do not treat stage-local layer 0 as block 0."""
    import re
    from rwkv_conversion import map_megatron_to_rwkv_key

    match = re.search(r'decoder\.layers\.(\d+)\.', name)
    global_name = name
    if match:
        local = int(match.group(1))
        layer = model.decoder.layers[local]
        global_id = layer.self_attention.layer_number - 1
        global_name = name[:match.start(1)] + str(global_id) + name[match.end(1):]
    native = map_megatron_to_rwkv_key(global_name, 0, 0)
    return [global_name] + (native if isinstance(native, list) else [native] if native else [])


def _decision(component, aliases, policy):
    decisions = []
    for alias in aliases:
        freeze = policy.get(component, False)
        if any(fnmatchcase(alias, p) for p in policy['freeze_extra_patterns']):
            freeze = True
        if any(fnmatchcase(alias, p) for p in policy['train_extra_patterns']):
            freeze = False
        decisions.append(freeze)
    # A match in either naming convention is sufficient; explicit train wins.
    train = any(fnmatchcase(a, p) for a in aliases for p in policy['train_extra_patterns'])
    return False if train else any(decisions)


def freeze_rwkv_distill_layers(model, freeze_embedding=False, freeze_lm_head=False, *, args=None):
    """Freeze weights only: autograd through frozen projections remains enabled."""
    if args is None:
        args = SimpleNamespace(rwkv_freeze_embedding=freeze_embedding,
                               rwkv_freeze_lm_head=freeze_lm_head)
    policy = freeze_policy(args)
    frozen = dict.fromkeys(FREEZE_COMPONENTS, 0)
    frozen['extra'] = 0
    if getattr(model, 'share_embeddings_and_output_weights', False):
        embedding = _decision('embedding', ['embedding.word_embeddings.weight', 'emb.weight'], policy)
        head = _decision('lm_head', ['output_layer.weight', 'head.weight'], policy)
        if embedding != head:
            raise ValueError('Tied embedding/head must use the same freeze policy; '
                             'otherwise set untie_embeddings_and_output_weights=true')
    for name, parameter in model.named_parameters():
        component = _component(name)
        aliases = parameter_aliases(model, name)
        freeze = _decision(component, aliases, policy)
        # Each of the six native attention shifts shares one Megatron parameter.
        if component == 'token_shift' and len(aliases) > 2:
            decisions = [_decision(component, [aliases[0], a], policy) for a in aliases[1:]]
            if len(set(decisions)) > 1:
                raise ValueError('Partial attention token-shift patterns cannot freeze a fused '
                                 'parameter; select all att.x_* or use rwkv_freeze_token_shift')
        explicit_train = any(fnmatchcase(a, p) for a in aliases
                             for p in policy['train_extra_patterns'])
        if freeze:
            parameter.requires_grad_(False)
            frozen[component or 'extra'] += parameter.numel()
        elif explicit_train:
            parameter.requires_grad_(True)
    return frozen


def check_freeze_checkpoint(args, checkpoint_args, *, release=False):
    """Never map Adam slots by position across different trainable parameter sets.

    --no-load-optim is an explicit reset of both Adam and its LR scheduler;
    normal resume still restores iteration, consumed samples and RNG. Finetune
    already opts into weights-only loading and keeps its upstream semantics.
    """
    changed = (freeze_policy(args) != freeze_policy(checkpoint_args)
               if checkpoint_args is not None else any(freeze_policy(args).values()))
    if changed and getattr(args, 'load_main_params_from_ckpt', False):
        raise ValueError('Freeze strategy changed: disable load_main_params_from_ckpt; '
                         'fresh FP32 master weights must be rebuilt from loaded model weights')
    if release or getattr(args, 'finetune', False) or getattr(args, 'no_load_optim', False):
        return
    if checkpoint_args is None:
        if any(freeze_policy(args).values()):
            raise ValueError('Checkpoint has no freeze metadata; use --no-load-optim '
                             'to reset Adam and LR scheduler safely')
        return
    if getattr(checkpoint_args, 'no_save_optim', False):
        return
    if freeze_policy(args) != freeze_policy(checkpoint_args):
        raise ValueError('RWKV freeze strategy differs from checkpoint: old Adam slots/master '
                         'weights must not be loaded into a different parameter partition. '
                         'Set no_load_optim: true (--no-load-optim) to reset Adam and LR '
                         'scheduler while retaining checkpoint iteration, consumed samples '
                         'and RNG; do not use finetune for an ordinary data-position resume.')


@contextmanager
def freeze_checkpoint_guard():
    """Check metadata before Megatron builds/loads sharded optimizer state.

    Scope the adapter to this entrypoint; no vendored Megatron modification.
    The first metadata read covers torch_dist, and the final read covers legacy.
    """
    from megatron.training import checkpointing
    original = checkpointing._load_base_checkpoint

    def guarded(load_dir, args, *positional, **kwargs):
        result = original(load_dir, args, *positional, **kwargs)
        state, _, release, _ = result
        if state is not None:
            check_freeze_checkpoint(args, state.get('args'), release=release)
        return result

    with patch.object(checkpointing, '_load_base_checkpoint', guarded):
        yield
