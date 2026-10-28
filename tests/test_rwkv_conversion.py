import contextlib
import copy
import io
import math
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import rwkv_conversion as converter


SOURCE_SUFFIXES = {
    'input_layernorm.weight': ('ln1.weight', -1),
    'input_layernorm.bias': ('ln1.bias', -1),
    'pre_mlp_layernorm.weight': ('ln2.weight', -1),
    'pre_mlp_layernorm.bias': ('ln2.bias', -1),
    'self_attention.rho': ('att.r_k', 0),
    'self_attention.beta': ('att.k_k', 0),
    'self_attention.alpha': ('att.k_a', 0),
    'self_attention.bottleneck_learn.weight_down': ('att.a1', -1),
    'self_attention.bottleneck_learn.weight_up': ('att.a2', -1),
    'self_attention.bottleneck_learn.bias': ('att.a0', 0),
    'self_attention.bottleneck_decay.weight_down': ('att.w1', -1),
    'self_attention.bottleneck_decay.weight_up': ('att.w2', -1),
    'self_attention.bottleneck_decay.bias': ('att.w0', 0),
    'self_attention.bottleneck_gate.weight_down': ('att.g1', -1),
    'self_attention.bottleneck_gate.weight_up': ('att.g2', -1),
    'self_attention.bottleneck_v.weight_down': ('att.v1', -1),
    'self_attention.bottleneck_v.weight_up': ('att.v2', -1),
    'self_attention.bottleneck_v.bias': ('att.v0', 0),
    'self_attention.linear_receptance.weight': ('att.receptance.weight', 0),
    'self_attention.linear_key.weight': ('att.key.weight', 0),
    'self_attention.linear_value.weight': ('att.value.weight', 0),
    'self_attention.linear_proj.weight': ('att.output.weight', 1),
    'self_attention.group_norm.weight': ('att.ln_x.weight', 0),
    'self_attention.group_norm.bias': ('att.ln_x.bias', 0),
    'mlp.mix_coeffs': ('ffn.x_k', -1),
    'mlp.linear_up.weight': ('ffn.key.weight', 0),
    'mlp.linear_down.weight': ('ffn.value.weight', 1),
}


def make_config(tp_size=2, pp_size=2, hidden_size=8, heads=4, num_layers=4,
                zero_centered=False):
    return Namespace(
        tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size,
        hidden_size=hidden_size, num_attention_heads=heads, num_layers=num_layers,
        padded_vocab_size=16, ffn_hidden_size=hidden_size * 4,
        D_AAA_LORA=3, D_DECAY_LORA=2, D_GATE_LORA=4, D_MV_LORA=2,
        layernorm_zero_centered_gamma=zero_centered,
        virtual_pipeline_model_parallel_size=None,
        num_layers_in_first_pipeline_stage=None, num_layers_in_last_pipeline_stage=None,
    )


def make_native_model(config, dtype=torch.bfloat16):
    hidden_size = config.hidden_size
    shapes = OrderedDict([
        ('emb.weight', (config.padded_vocab_size, hidden_size)),
        ('head.weight', (config.padded_vocab_size, hidden_size)),
        ('ln_out.weight', (hidden_size,)), ('ln_out.bias', (hidden_size,)),
        ('blocks.0.ln0.weight', (hidden_size,)), ('blocks.0.ln0.bias', (hidden_size,)),
    ])
    for layer in range(config.num_layers):
        prefix = f'blocks.{layer}.'
        for norm in ('ln1', 'ln2'):
            shapes[prefix + norm + '.weight'] = (hidden_size,)
            shapes[prefix + norm + '.bias'] = (hidden_size,)
        for channel in 'rwkvag':
            shapes[prefix + 'att.x_' + channel] = (1, 1, hidden_size)
        for vector in ('k_k', 'k_a', 'w0', 'a0'):
            shapes[prefix + 'att.' + vector] = (1, 1, hidden_size)
        shapes[prefix + 'att.r_k'] = (config.num_attention_heads,
                                    hidden_size // config.num_attention_heads)
        for family, rank in [('w', 2), ('a', 3), ('g', 4), ('v', 2)]:
            if family == 'v' and layer == 0:
                continue
            shapes[prefix + f'att.{family}1'] = (hidden_size, rank)
            shapes[prefix + f'att.{family}2'] = (rank, hidden_size)
            if family == 'v':
                shapes[prefix + 'att.v0'] = (1, 1, hidden_size)
        for projection in ('receptance', 'key', 'value', 'output'):
            shapes[prefix + f'att.{projection}.weight'] = (hidden_size, hidden_size)
        shapes[prefix + 'att.ln_x.weight'] = (hidden_size,)
        shapes[prefix + 'att.ln_x.bias'] = (hidden_size,)
        shapes[prefix + 'ffn.x_k'] = (1, 1, hidden_size)
        shapes[prefix + 'ffn.key.weight'] = (config.ffn_hidden_size, hidden_size)
        shapes[prefix + 'ffn.value.weight'] = (hidden_size, config.ffn_hidden_size)
    return OrderedDict(
        (key, ((torch.arange(math.prod(shape)).reshape(shape) + index * 3) % 31 / 64).to(dtype))
        for index, (key, shape) in enumerate(shapes.items())
    )


def make_shards(native, config, pp_rank):
    counts = converter.pipeline_layer_counts(config)
    layers_per_stage = counts[pp_rank]
    full_stage = OrderedDict()
    split_dims = {}

    def add(source, native_key, dim):
        tensor = native[native_key]
        if source.endswith(('.rho', '.beta', '.alpha', '.bias', '.mix_coeffs')):
            tensor = tensor.flatten()
        if config.layernorm_zero_centered_gamma and (
            native_key == 'ln_out.weight' or
            any(native_key.endswith(f'.ln{norm}.weight') for norm in range(3))
        ):
            tensor = tensor - 1
        full_stage[source] = tensor
        split_dims[source] = dim

    if pp_rank == 0:
        add('embedding.word_embeddings.weight', 'emb.weight', 0)
        add('pre_norm.weight', 'blocks.0.ln0.weight', -1)
        add('pre_norm.bias', 'blocks.0.ln0.bias', -1)
    if pp_rank == config.pipeline_model_parallel_size - 1:
        add('output_layer.weight', 'head.weight', 0)
        add('decoder.final_layernorm.weight', 'ln_out.weight', -1)
        add('decoder.final_layernorm.bias', 'ln_out.bias', -1)
    for local_layer in range(layers_per_stage):
        global_layer = sum(counts[:pp_rank]) + local_layer
        source_prefix = f'decoder.layers.{local_layer}.'
        native_prefix = f'blocks.{global_layer}.'
        for suffix, (native_suffix, dim) in SOURCE_SUFFIXES.items():
            if global_layer == 0 and suffix.startswith('self_attention.bottleneck_v.'):
                shape = {
                    'weight_down': (config.hidden_size, config.D_MV_LORA),
                    'weight_up': (config.D_MV_LORA, config.hidden_size),
                    'bias': (config.hidden_size,),
                }[suffix.rsplit('.', 1)[1]]
                full_stage[source_prefix + suffix] = torch.zeros(shape, dtype=next(iter(native.values())).dtype)
                split_dims[source_prefix + suffix] = dim
            else:
                add(source_prefix + suffix, native_prefix + native_suffix, dim)
        mix_key = source_prefix + 'self_attention.mix_coeff_input'
        full_stage[mix_key] = torch.cat([native[native_prefix + f'att.x_{channel}'] for channel in 'rwkvag']).reshape(6, config.hidden_size)
        split_dims[mix_key] = -1
        full_stage[source_prefix + 'self_attention.linear_key._extra_state'] = None
    shards = []
    for tp_rank in range(config.tensor_model_parallel_size):
        model = OrderedDict()
        for key, tensor in full_stage.items():
            dim = split_dims.get(key, -1)
            if isinstance(tensor, torch.Tensor) and dim >= 0:
                tensor = tensor.chunk(config.tensor_model_parallel_size, dim=dim)[tp_rank]
            model[key] = tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
        shards.append(model)
    return shards


def write_checkpoint(root, config, native, iteration=7):
    iteration_dir = Path(root) / f'iter_{iteration:07d}'
    for pp_rank in range(config.pipeline_model_parallel_size):
        for tp_rank, model in enumerate(make_shards(native, config, pp_rank)):
            rank_name = f'mp_rank_{tp_rank:02d}'
            if config.pipeline_model_parallel_size > 1:
                rank_name += f'_{pp_rank:03d}'
            directory = iteration_dir / rank_name
            directory.mkdir(parents=True, exist_ok=True)
            torch.save({'args': config, 'iteration': iteration, 'model': model},
                       directory / 'model_optim_rng.pt')
    (Path(root) / 'latest_checkpointed_iteration.txt').write_text(str(iteration))
    return iteration_dir


class ConversionTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = make_config()
        self.native = make_native_model(self.config)
        self.source = self.root / 'source'
        self.output = self.root / 'output'
        self.iteration_dir = write_checkpoint(self.source, self.config, self.native)

    def run_conversion(self, load_dir=None, **options):
        args = Namespace(load_dir=str(load_dir or self.source), save_dir=str(self.output),
                         iteration=options.get('iteration'), norm_offset=options.get('norm_offset'))
        with contextlib.redirect_stdout(io.StringIO()):
            path = converter.main(args)
        return torch.load(path, map_location='cpu', weights_only=True)

    def assert_model_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key, tensor in expected.items():
            self.assertEqual(actual[key].dtype, tensor.dtype, key)
            self.assertEqual(actual[key].shape, tensor.shape, key)
            self.assertTrue(torch.equal(actual[key], tensor), key)
            self.assertTrue(actual[key].is_contiguous(), key)
            self.assertEqual(actual[key].untyped_storage().nbytes(),
                             actual[key].numel() * actual[key].element_size(), key)

    def edit_shard(self, mutate, tp_rank=0, pp_rank=0):
        path = self.iteration_dir / f'mp_rank_{tp_rank:02d}_{pp_rank:03d}' / 'model_optim_rng.pt'
        data = torch.load(path, weights_only=False)
        mutate(data)
        torch.save(data, path)

    def assert_rejected_without_saving(self, expression):
        save_dir = self.output / 'iter_0000007'
        save_dir.mkdir(parents=True)
        save_file = save_dir / 'rwkv_state_dict.pth'
        save_file.write_bytes(b'previous valid model')
        tracker = self.output / 'latest_checkpointed_iteration.txt'
        tracker.write_text('6')
        with patch.object(converter.torch, 'save') as save:
            with self.assertRaisesRegex((ValueError, RuntimeError, FileNotFoundError), expression):
                self.run_conversion()
            save.assert_not_called()
        self.assertEqual(save_file.read_bytes(), b'previous valid model')
        self.assertEqual(tracker.read_text(), '6')


class CheckpointSelectionTests(ConversionTestCase):
    def test_default_tracker_roundtrip(self):
        self.assert_model_equal(self.run_conversion(), self.native)
        self.assertEqual((self.output / 'latest_checkpointed_iteration.txt').read_text(), '7')

    def test_explicit_iteration_ignores_tracker(self):
        (self.source / 'latest_checkpointed_iteration.txt').write_text('999')
        self.assert_model_equal(self.run_conversion(iteration=7), self.native)

    def test_explicit_iteration_without_tracker(self):
        (self.source / 'latest_checkpointed_iteration.txt').unlink()
        self.assert_model_equal(self.run_conversion(iteration=7), self.native)

    def test_direct_iteration_directory_with_trailing_slash(self):
        (self.source / 'latest_checkpointed_iteration.txt').write_text('999')
        self.assert_model_equal(self.run_conversion(str(self.iteration_dir) + '/'), self.native)

    def test_matching_direct_iteration(self):
        self.assert_model_equal(self.run_conversion(self.iteration_dir, iteration=7), self.native)

    def test_conflicting_direct_iteration(self):
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            self.run_conversion(self.iteration_dir, iteration=8)
        self.assertFalse(self.output.exists())

    def test_invalid_tracker(self):
        (self.source / 'latest_checkpointed_iteration.txt').write_text('release')
        self.assert_rejected_without_saving('Invalid iteration')

    def test_missing_tracker(self):
        (self.source / 'latest_checkpointed_iteration.txt').unlink()
        self.assert_rejected_without_saving('latest_checkpointed_iteration')

    def test_negative_iteration(self):
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            self.run_conversion(iteration=-1)

    def test_missing_iteration_directory(self):
        with self.assertRaisesRegex(FileNotFoundError, 'directory not found'):
            self.run_conversion(iteration=8)

    def test_cli_direct_directory(self):
        result = subprocess.run([
            sys.executable, str(REPO_ROOT / 'rwkv_conversion.py'),
            '--load-dir', str(self.iteration_dir), '--iteration', '7',
            '--save-dir', str(self.output), '--norm-offset', '0',
        ], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = torch.load(self.output / 'iter_0000007' / 'rwkv_state_dict.pth', weights_only=True)
        self.assert_model_equal(actual, self.native)


class NormOffsetTests(ConversionTestCase):
    def test_explicit_offset_only_changes_layernorm_gamma(self):
        actual = self.run_conversion(norm_offset=1.0)
        for key, tensor in self.native.items():
            expected = tensor + 1 if key == 'ln_out.weight' or any(
                key.endswith(f'.ln{norm}.weight') for norm in range(3)
            ) else tensor
            self.assertTrue(torch.equal(actual[key], expected), key)

    def test_zero_centered_gamma_automatically_restores_weights(self):
        self.config.layernorm_zero_centered_gamma = True
        write_checkpoint(self.source, self.config, self.native)
        actual = self.run_conversion()
        self.assert_model_equal(actual, self.native)
        inputs = torch.arange(16).reshape(2, 8).float()
        for prefix in ['blocks.0.ln0', 'blocks.0.ln1', 'blocks.2.ln2', 'ln_out']:
            expected = torch.nn.functional.layer_norm(
                inputs, (8,), self.native[prefix + '.weight'].float(),
                self.native[prefix + '.bias'].float(),
            )
            restored = torch.nn.functional.layer_norm(
                inputs, (8,), actual[prefix + '.weight'].float(), actual[prefix + '.bias'].float(),
            )
            self.assertTrue(torch.equal(restored, expected), prefix)

    def test_explicit_zero_overrides_automatic_offset(self):
        self.config.layernorm_zero_centered_gamma = True
        write_checkpoint(self.source, self.config, self.native)
        actual = self.run_conversion(norm_offset=0.0)
        expected = copy.deepcopy(self.native)
        for key in expected:
            if key == 'ln_out.weight' or any(key.endswith(f'.ln{norm}.weight') for norm in range(3)):
                expected[key] -= 1
        self.assert_model_equal(actual, expected)

    def test_custom_negative_offset(self):
        actual = self.run_conversion(norm_offset=-0.5)
        self.assertTrue(torch.equal(actual['ln_out.weight'], self.native['ln_out.weight'] - 0.5))
        self.assertTrue(torch.equal(actual['ln_out.bias'], self.native['ln_out.bias']))

    def test_nonfinite_offset_rejected(self):
        for offset in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, 'must be finite'):
                self.run_conversion(norm_offset=offset)
        self.assertFalse(self.output.exists())


class ValidationTests(ConversionTestCase):
    def test_missing_critical_parameters_rejected_before_save(self):
        for key in ['blocks.0.att.key.weight', 'blocks.3.att.v2', 'blocks.2.att.x_g',
                    'blocks.0.ln0.bias', 'ln_out.weight', 'head.weight']:
            with self.subTest(key=key):
                invalid = self.native.copy()
                invalid.pop(key)
                with self.assertRaisesRegex(ValueError, 'missing|Missing'):
                    converter.validate_rwkv_state_dict(invalid, self.config)
        for tp_rank in range(2):
            self.edit_shard(lambda data: data['model'].pop('decoder.layers.0.self_attention.linear_key.weight'), tp_rank)
        self.assert_rejected_without_saving('missing=')

    def test_invalid_shapes(self):
        for key in ['emb.weight', 'head.weight', 'ln_out.bias', 'blocks.0.att.r_k',
                    'blocks.1.att.v2', 'blocks.0.ffn.value.weight', 'blocks.0.att.x_w']:
            with self.subTest(key=key):
                invalid = self.native.copy()
                invalid[key] = invalid[key].flatten()[:1]
                with self.assertRaisesRegex(ValueError, 'Invalid shape'):
                    converter.validate_rwkv_state_dict(invalid, self.config)

    def test_wrong_head_shape_rejected(self):
        invalid = self.native.copy()
        invalid['blocks.0.att.r_k'] = invalid['blocks.0.att.r_k'].reshape(2, 4)
        with self.assertRaisesRegex(ValueError, 'Invalid shape.*r_k'):
            converter.validate_rwkv_state_dict(invalid, self.config)

    def test_nonfinite_parameter_rejected_before_save(self):
        self.edit_shard(lambda data: data['model']['embedding.word_embeddings.weight'].fill_(float('nan')))
        self.assert_rejected_without_saving('Nonfinite')

    def test_nonfloating_parameter_rejected(self):
        invalid = self.native.copy()
        invalid['head.weight'] = invalid['head.weight'].long()
        with self.assertRaisesRegex(ValueError, 'floating-point'):
            converter.validate_rwkv_state_dict(invalid, self.config)

    def test_unexpected_output_key_rejected(self):
        invalid = self.native.copy()
        invalid['blocks.99.att.r_k'] = torch.zeros(4, 2)
        with self.assertRaisesRegex(ValueError, 'unexpected='):
            converter.validate_rwkv_state_dict(invalid, self.config)

    def test_unmapped_source_key_rejected(self):
        for tp_rank in range(2):
            self.edit_shard(lambda data: data['model'].update({'decoder.layers.0.unknown.weight': torch.zeros(8, dtype=torch.bfloat16)}), tp_rank)
        self.assert_rejected_without_saving('Unmapped')

    def test_tp_key_mismatch_rejected(self):
        self.edit_shard(lambda data: data['model'].pop('decoder.layers.0.input_layernorm.weight'))
        self.assert_rejected_without_saving('TP key mismatch')

    def test_tp_replicated_value_mismatch_rejected(self):
        self.edit_shard(lambda data: data['model']['decoder.layers.0.input_layernorm.weight'].add_(1))
        self.assert_rejected_without_saving('Replicated TP tensors differ')

    def test_tp_shape_mismatch_rejected(self):
        self.edit_shard(lambda data: data['model'].update({'embedding.word_embeddings.weight': torch.zeros(8, 7, dtype=torch.bfloat16)}))
        self.assert_rejected_without_saving('TP shape mismatch')

    def test_tp_dtype_mismatch_rejected(self):
        self.edit_shard(lambda data: data['model'].update({'embedding.word_embeddings.weight': data['model']['embedding.word_embeddings.weight'].float()}))
        self.assert_rejected_without_saving('TP dtype mismatch')

    def test_missing_rank_rejected(self):
        (self.iteration_dir / 'mp_rank_01_001' / 'model_optim_rng.pt').unlink()
        self.assert_rejected_without_saving('model_optim_rng')

    def test_iteration_metadata_mismatch_rejected(self):
        self.edit_shard(lambda data: data.update({'iteration': 8}))
        self.assert_rejected_without_saving('iteration mismatch')

    def test_shard_config_mismatch_rejected(self):
        self.edit_shard(lambda data: setattr(data['args'], 'num_attention_heads', 2), tp_rank=1)
        self.assert_rejected_without_saving('config mismatch')

    def test_invalid_local_layer_rejected(self):
        for tp_rank in range(2):
            self.edit_shard(lambda data: data['model'].update({'decoder.layers.2.input_layernorm.weight': torch.zeros(8, dtype=torch.bfloat16)}), tp_rank)
        self.assert_rejected_without_saving('Invalid local pipeline layer')

    def test_duplicate_pipeline_output_rejected(self):
        for tp_rank in range(2):
            self.edit_shard(lambda data: data['model'].update({'pre_norm.weight': torch.zeros(8, dtype=torch.bfloat16)}), tp_rank, pp_rank=1)
        self.assert_rejected_without_saving('Duplicate output parameters')

    def test_invalid_mix_rows_rejected(self):
        for tp_rank in range(2):
            self.edit_shard(lambda data: data['model'].update({'decoder.layers.0.self_attention.mix_coeff_input': torch.zeros(7, 8, dtype=torch.bfloat16)}), tp_rank)
        self.assert_rejected_without_saving('Invalid mix_coeff_input shape')

    def test_unsupported_configs_rejected(self):
        for name, value in [('virtual_pipeline_model_parallel_size', 2),
                            ('num_layers_in_first_pipeline_stage', 0),
                            ('num_layers_in_last_pipeline_stage', 4),
                            ('decoder_first_pipeline_num_layers', True),
                            ('decoder_last_pipeline_num_layers', -1),
                            ('pipeline_model_parallel_layout', 'Et|tt|tL'),
                            ('num_layers', 3), ('num_attention_heads', 3),
                            ('tensor_model_parallel_size', 8), ('hidden_size', 0)]:
            with self.subTest(name=name):
                config = copy.deepcopy(self.config)
                setattr(config, name, value)
                with self.assertRaises(ValueError):
                    converter.validate_model_config(config)

    def test_uneven_pipeline_native_roundtrip_and_saved_arg_aliases(self):
        for first_name, last_name in [
            ('num_layers_in_first_pipeline_stage', 'num_layers_in_last_pipeline_stage'),
            ('decoder_first_pipeline_num_layers', 'decoder_last_pipeline_num_layers'),
        ]:
            with self.subTest(first_name=first_name):
                config = make_config(pp_size=4, num_layers=12)
                setattr(config, first_name, 2)
                setattr(config, last_name, 2)
                converter.validate_model_config(config)
                self.assertEqual(converter.pipeline_layer_counts(config), [2, 4, 4, 2])
                native = make_native_model(config)
                merged = OrderedDict()
                for stage in range(4):
                    part = converter.convert_pipeline_stage(make_shards(native, config, stage), stage, config, 0)
                    self.assertFalse(merged.keys() & part.keys())
                    merged.update(part)
                self.assertEqual(set(merged), set(native))
                for key in native:
                    self.assertTrue(torch.equal(merged[key], native[key]), key)

    def test_conflicting_pipeline_aliases_rejected(self):
        config = make_config()
        config.num_layers_in_first_pipeline_stage = 1
        config.decoder_first_pipeline_num_layers = 2
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            converter.validate_model_config(config)

    def test_atomic_write_failure_preserves_previous_artifacts(self):
        save_dir = self.output / 'iter_0000007'
        save_dir.mkdir(parents=True)
        save_file = save_dir / 'rwkv_state_dict.pth'
        save_file.write_bytes(b'previous model')
        tracker = self.output / 'latest_checkpointed_iteration.txt'
        tracker.write_text('6')

        def fail_save(state_dict, path):
            Path(path).write_bytes(b'partial model')
            raise OSError('simulated disk failure')

        with patch.object(converter.torch, 'save', side_effect=fail_save):
            with self.assertRaisesRegex(OSError, 'disk failure'):
                self.run_conversion()
        self.assertEqual(save_file.read_bytes(), b'previous model')
        self.assertEqual(tracker.read_text(), '6')
        self.assertEqual(list(save_dir.iterdir()), [save_file])


class ParallelRegressionTests(ConversionTestCase):
    def test_standard_30b_shape_contract(self):
        config = make_config(tp_size=4, pp_size=2, hidden_size=6144, heads=96, num_layers=64)
        converter.validate_model_config(config)
        shapes = converter.expected_rwkv_shapes(config, {})
        self.assertEqual(len(shapes), 2115)
        self.assertEqual(shapes['blocks.0.att.r_k'], (96, 64))
        self.assertEqual(shapes['blocks.63.att.r_k'], (96, 64))
        self.assertEqual(shapes['blocks.63.ffn.key.weight'], (24576, 6144))
        self.assertNotIn('blocks.0.att.v0', shapes)

    def test_tp_pp_dtype_roundtrips(self):
        for tp_size, pp_size, dtype in [(1, 1, torch.float32), (2, 1, torch.float16),
                                        (1, 2, torch.bfloat16), (4, 2, torch.bfloat16)]:
            with self.subTest(tp=tp_size, pp=pp_size, dtype=dtype):
                config = make_config(tp_size=tp_size, pp_size=pp_size)
                native = make_native_model(config, dtype)
                source = self.root / f'tp{tp_size}-pp{pp_size}'
                write_checkpoint(source, config, native)
                self.assert_model_equal(self.run_conversion(source), native)

    def test_standard_head_size_64_roundtrip(self):
        config = make_config(tp_size=4, hidden_size=256, heads=4)
        native = make_native_model(config)
        source = self.root / 'standard-head64'
        write_checkpoint(source, config, native)
        actual = self.run_conversion(source)
        self.assertEqual(actual['blocks.0.att.r_k'].shape, (4, 64))
        self.assert_model_equal(actual, native)

    def test_nonstandard_head_size_is_preserved(self):
        config = make_config(tp_size=2, hidden_size=192, heads=2)
        native = make_native_model(config)
        source = self.root / 'head96'
        write_checkpoint(source, config, native)
        actual = self.run_conversion(source)
        self.assertEqual(actual['blocks.0.att.r_k'].shape, (2, 96))
        self.assert_model_equal(actual, native)

    def test_single_layer_model(self):
        config = make_config(tp_size=1, pp_size=1, num_layers=1)
        native = make_native_model(config)
        source = self.root / 'single-layer'
        write_checkpoint(source, config, native)
        self.assert_model_equal(self.run_conversion(source), native)

    def test_legacy_config_infers_vocab_ffn_and_lora_shapes(self):
        config = copy.deepcopy(self.config)
        for name in ['padded_vocab_size', 'ffn_hidden_size', 'D_AAA_LORA',
                     'D_DECAY_LORA', 'D_GATE_LORA', 'D_MV_LORA']:
            delattr(config, name)
        converter.validate_rwkv_state_dict(self.native, config)

    def test_compact_storage_for_replicated_shard(self):
        storage = torch.arange(10000).float()
        compact = converter.combine_tp_tensors('norm', -1, [storage[100:104]])
        self.assertEqual(compact.untyped_storage().nbytes(), 16)


if __name__ == '__main__':
    unittest.main()
