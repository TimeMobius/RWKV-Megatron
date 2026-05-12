"""
Post-fix regression test for RWKV7 bottleneck TP batch mixing.

Expected workflow:
1. Run this script before changing rwkv7_layer_specs.py. It should fail on the current code.
2. After fixing the TP reshape / communication logic, this script should pass and confirm
   that hidden sharding no longer mixes different batch samples.
"""

import torch

from rwkv7_bottleneck_batchmix_test_utils import (
    assert_tensor_equal,
    expected_hidden_shards,
    run_non_sequence_parallel_outputs,
    run_sequence_parallel_outputs,
)


def test_non_sequence_parallel_keeps_samples_isolated():
    multi_batch = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [100.0, 101.0, 102.0, 103.0]]])
    outputs = run_non_sequence_parallel_outputs(multi_batch)
    expected = expected_hidden_shards(multi_batch)
    assert_tensor_equal("non-SP rank0", outputs[0], expected[0])
    assert_tensor_equal("non-SP rank1", outputs[1], expected[1])


def test_sequence_parallel_keeps_samples_isolated():
    global_x = torch.tensor(
        [
            [[0.0, 1.0, 2.0, 3.0], [100.0, 101.0, 102.0, 103.0]],
            [[1000.0, 1001.0, 1002.0, 1003.0], [1100.0, 1101.0, 1102.0, 1103.0]],
        ]
    )
    outputs = run_sequence_parallel_outputs(global_x)
    expected = expected_hidden_shards(global_x)
    assert_tensor_equal("SP rank0", outputs[0], expected[0])
    assert_tensor_equal("SP rank1", outputs[1], expected[1])


if __name__ == "__main__":
    test_non_sequence_parallel_keeps_samples_isolated()
    test_sequence_parallel_keeps_samples_isolated()
    print("PASS: RWKV7 bottleneck TP sharding keeps batch samples isolated.")
