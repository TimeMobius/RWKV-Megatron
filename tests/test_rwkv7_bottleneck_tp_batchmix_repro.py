"""
Pre-fix repro for RWKV7 bottleneck TP batch mixing.

Expected workflow:
1. Run this script before changing rwkv7_layer_specs.py. It should pass and confirm
   that the current implementation mixes hidden channels across different batch samples.
2. After the code fix, this script should fail, because the buggy contamination pattern
   should disappear.
"""

import pytest
import torch

from rwkv7_bottleneck_batchmix_test_utils import (
    expected_hidden_shards,
    run_non_sequence_parallel_outputs,
    run_sequence_parallel_outputs,
)


@pytest.mark.xfail(strict=True, reason="Historical pre-fix bug reproduction; fixed code must fail this assertion")
def test_non_sequence_parallel_reproduces_batch_mixing():
    single_batch = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
    multi_batch = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [100.0, 101.0, 102.0, 103.0]]])

    single_outputs = run_non_sequence_parallel_outputs(single_batch)
    single_expected = expected_hidden_shards(single_batch)
    assert torch.equal(single_outputs[0], single_expected[0])
    assert torch.equal(single_outputs[1], single_expected[1])

    multi_outputs = run_non_sequence_parallel_outputs(multi_batch)
    multi_expected = expected_hidden_shards(multi_batch)

    assert not torch.equal(multi_outputs[0], multi_expected[0])
    assert not torch.equal(multi_outputs[1], multi_expected[1])
    assert multi_outputs[0][0, 1].tolist() == [2.0, 3.0]
    assert multi_expected[0][0, 1].tolist() == [100.0, 101.0]


@pytest.mark.xfail(strict=True, reason="Historical pre-fix bug reproduction; fixed code must fail this assertion")
def test_sequence_parallel_reproduces_batch_mixing():
    global_x = torch.tensor(
        [
            [[0.0, 1.0, 2.0, 3.0], [100.0, 101.0, 102.0, 103.0]],
            [[1000.0, 1001.0, 1002.0, 1003.0], [1100.0, 1101.0, 1102.0, 1103.0]],
        ]
    )

    outputs = run_sequence_parallel_outputs(global_x)
    expected = expected_hidden_shards(global_x)

    assert not torch.equal(outputs[0], expected[0])
    assert not torch.equal(outputs[1], expected[1])
    assert outputs[0][0, 1].tolist() == [2.0, 3.0]
    assert expected[0][0, 1].tolist() == [100.0, 101.0]


if __name__ == "__main__":
    test_non_sequence_parallel_reproduces_batch_mixing()
    test_sequence_parallel_reproduces_batch_mixing()
    print("PASS: current RWKV7 bottleneck implementation reproduces the TP batch mixing bug.")
