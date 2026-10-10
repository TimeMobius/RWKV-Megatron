"""Regression checks for RWKV state transport across upstream API boundaries."""

import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.models.gpt import GPTModel
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.utils import Split
from megatron.core.pipeline_parallel.schedules import get_tensor_shapes
from megatron.core.tokenizers.text.libraries.huggingface_tokenizer import HuggingFaceTokenizer
from rwkv_megatron.rwkv7_model import RWKV7Model
from rwkv_megatron.rwkv7_transformer_layer import RWKV7TransformerLayer


class ToyLayer(RWKV7TransformerLayer):
    """Exercise the real carrier forward with simple differentiable submodules."""

    def __init__(self, first):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(cuda_graph_impl="none")
        self.first = first
        self.weight = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))

    def _run_input_layernorm(self, hidden):
        return hidden * 2, hidden, ()

    def self_attention(self, hidden, **kwargs):
        if self.first:
            values = hidden * self.weight
        else:
            values = hidden[1]
            hidden = hidden[0]
        output = hidden * self.weight + values
        return torch.stack((output, values)), None

    def _apply_self_attn_bda_step(self, output, residual, state):
        return output[0] + residual

    def _run_cross_attention(self, hidden, context, mask, inference):
        return hidden, context

    def _forward_mlp(self, hidden, *args, **kwargs):
        return hidden * 3


class CompatibilityTests(unittest.TestCase):
    def test_exhaustive_blend_shuffle_preserves_samples_and_cached_order(self):
        class Dataset:
            index_split = Split.train

            def __init__(self, source):
                self.unique_identifiers = {"source": source}

            def __len__(self):
                return 64

            def __getitem__(self, sample):
                return {"sample": int(sample)}

        with TemporaryDirectory() as cache:
            datasets = [Dataset(0), Dataset(1)]

            def build(shuffle, seed=1234, size=None):
                config = SimpleNamespace(
                    shuffle_blend_datasets=shuffle, random_seed=seed,
                    path_to_cache=cache, defer_npy_index_mmap=False, fast_cache_load=False,
                )
                return BlendedDataset(datasets, [17, 23], size, config)

            original, shuffled = build(False), build(True)
            cached = build(True)
            pairs = lambda ds: [(int(ds[i]["dataset_id"]), ds[i]["sample"]) for i in range(len(ds))]
            self.assertCountEqual(pairs(original), pairs(shuffled))
            self.assertNotEqual(pairs(original), pairs(shuffled))
            self.assertEqual(pairs(shuffled), pairs(cached))
            self.assertNotEqual(original.unique_description_hash, shuffled.unique_description_hash)
            # Rebuild without cache to check reproducibility, then vary the seed.
            shuffled.config.path_to_cache = None
            fresh_index, fresh_samples = shuffled._build_indices()
            np.testing.assert_array_equal(fresh_index, shuffled.dataset_index)
            np.testing.assert_array_equal(fresh_samples, shuffled.dataset_sample_index)
            shuffled.config.random_seed += 1
            other_index, other_samples = shuffled._build_indices()
            self.assertFalse(np.array_equal(np.stack((fresh_index, fresh_samples)),
                                            np.stack((other_index, other_samples))))
            self.assertEqual(pairs(build(False, size=20)), pairs(build(True, size=20)))

    def test_bos_is_preserved_when_legacy_tokenizer_aliases_pad_to_bos(self):
        class Dataset(MegatronDataset):
            @staticmethod
            def _key_config_attributes():
                return ()

            def __len__(self):
                return 0

            def __getitem__(self, index):
                raise IndexError(index)

        for pad in (65530, 65531, 65532):
            config = SimpleNamespace(
                tokenizer=SimpleNamespace(pad=pad, bos=65532, eos=65531, eod=65531),
                allow_ambiguous_pad_tokens=False,
            )
            if pad == 65530:
                dataset = Dataset(None, None, [], 0, Split.train, config)
                self.assertEqual(dataset._pad_token_id, pad)
            else:
                with self.assertWarnsRegex(UserWarning, "collides"):
                    dataset = Dataset(None, None, [], 0, Split.train, config)
                self.assertEqual(dataset._pad_token_id, -1)

    def test_value_residual_and_gradient_cross_layer_and_pipeline_boundary(self):
        source = (torch.arange(80, dtype=torch.float64).reshape(5, 2, 8) / 8).requires_grad_()
        first, second = ToyLayer(True), ToyLayer(False)
        carrier, _ = first(source)
        output, _ = second(carrier.clone())
        actual_grads = torch.autograd.grad(output[0].sum(), (source, first.weight, second.weight))

        values = source * 2 * first.weight
        hidden = (source * 2 * first.weight + values + source) * 3
        expected = (hidden * 2 * second.weight + values + hidden) * 3
        expected_grads = torch.autograd.grad(expected.sum(), (source, first.weight, second.weight))
        torch.testing.assert_close(output[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(output[1], values, rtol=0, atol=0)
        for actual, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected_grad, rtol=0, atol=0)

    def test_only_final_stage_removes_value_carrier(self):
        carrier = torch.randn(2, 5, 2, 8)
        for final in (False, True):
            model = RWKV7Model.__new__(RWKV7Model)
            torch.nn.Module.__init__(model)
            model.post_process = final
            with patch.object(GPTModel, "_postprocess", side_effect=lambda hidden, *a, **k: hidden):
                actual = model._postprocess(carrier)
            self.assertEqual(actual.shape, carrier[0].shape if final else carrier.shape)
            torch.testing.assert_close(actual, carrier[0] if final else carrier, rtol=0, atol=0)

    def test_pipeline_shapes_include_full_value_cache(self):
        for tp in (1, 2, 4):
            for sequence_parallel in (False, True):
                for rwkv in (False, True):
                    config = SimpleNamespace(
                        variable_seq_lengths=False, hidden_size=2560,
                        sequence_parallel=sequence_parallel, rwkv7_value_residual=rwkv,
                    )
                    shape = (4096 // tp if sequence_parallel else 4096, 1, 2560)
                    if rwkv:
                        shape = (1 + tp if sequence_parallel else 2, *shape)
                    actual = get_tensor_shapes(
                        seq_length=4096, micro_batch_size=1, decoder_seq_length=None,
                        config=config, tp_group=SimpleNamespace(size=lambda: tp),
                        cp_group=SimpleNamespace(size=lambda: 1),
                    )
                    self.assertEqual(actual, [shape])

    def test_duplicate_token_ids_do_not_expand_embedding_size(self):
        tokenizer = HuggingFaceTokenizer.__new__(HuggingFaceTokenizer)
        tokenizer.tokenizer = SimpleNamespace(
            get_vocab=lambda: {"text": 1, "last": 65534, "special": 65534}
        )
        self.assertEqual(tokenizer.vocab_size, 65535)
        self.assertEqual((tokenizer.vocab_size + 127) // 128 * 128, 65536)


if __name__ == "__main__":
    unittest.main()
