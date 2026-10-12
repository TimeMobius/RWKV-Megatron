"""Keep RWKV elementwise coefficients on Adam, including packed 2D vectors."""

import torch

from megatron.core.optimizer.emerging_optimizers import _is_muon_excluded
from megatron.core.optimizer.layer_wise_optimizer import is_managed_by_layer_wise_optimizer
from test_rwkv7_timemix_scratch_init_regression import _build_timemix


def test_timemix_coefficients_use_scalar_optimizer():
    module = _build_timemix(8, 2, 4, 1, 1, 0, 0)
    for name in ("mix_coeff_input", "rho", "beta", "alpha"):
        param = getattr(module, name)
        assert _is_muon_excluded(param), name
        assert not is_managed_by_layer_wise_optimizer(param), name
    for bottleneck in (module.bottleneck_decay, module.bottleneck_learn,
                       module.bottleneck_gate, module.bottleneck_v):
        for param in (bottleneck.weight_down, bottleneck.weight_up):
            assert not _is_muon_excluded(param)
            assert is_managed_by_layer_wise_optimizer(param)


def test_packed_coefficients_adam_update_matches_separate_vectors():
    module = _build_timemix(8, 2, 4, 1, 1, 0, 0)
    packed = module.mix_coeff_input
    vectors = [torch.nn.Parameter(row.detach().clone()) for row in packed]
    packed_optimizer = torch.optim.AdamW([packed], lr=1.56e-5, weight_decay=0.1)
    vector_optimizer = torch.optim.AdamW(vectors, lr=1.56e-5, weight_decay=0.1)
    generator = torch.Generator().manual_seed(1234)
    for _ in range(3):
        packed.grad = torch.randn(packed.shape, generator=generator)
        for vector, grad in zip(vectors, packed.grad):
            vector.grad = grad.clone()
        packed_optimizer.step()
        vector_optimizer.step()
        torch.testing.assert_close(packed, torch.stack(vectors), rtol=0, atol=0)
