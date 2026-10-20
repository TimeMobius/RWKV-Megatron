import pytest

from tools.backfill_distill_metrics import parse_metrics


LOG = """[Rank 0] (after 198 iterations) memory (MB) | allocated: 48305.32 | max allocated: 49397.14 | reserved: 49490.00 | max reserved: 49490.00
 [2026-10-16 17:16:51.000000] iteration      198/    1000 | consumed samples:          198 | elapsed time per iteration (ms): 2000.0 | throughput per GPU (TFLOP/s/GPU): 0.8 | learning rate: 1.000000E-06 | global batch size:     2 | lm loss: 1.400000E+01 | loss scale: 1.0 | grad norm: 24.067 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2026-10-16 17:16:52.000000] iteration      199/    1000 | consumed samples:          200 | elapsed time per iteration (ms): 1000.0 | throughput per GPU (TFLOP/s/GPU): 1.6 | learning rate: 1.000000E-06 | global batch size:     2 | lm loss: 1.300000E+01 | loss scale: 1.0 | grad norm: 10.276 | number of skipped iterations:   1 | number of nan iterations:   0 |
async Qwen teacher final stats: {'submitted': 16, 'consumed': 16, 'completed': 16, 'failed': 0, 'active': 0, 'peak_active': 16, 'buffered': 0}
"""


def test_backfill_units_axis_sparse_memory_and_final_teacher_snapshot():
    rows, teacher = parse_metrics(LOG, {'seq_length': 128})
    assert rows[0]['train/iteration'] == 198
    assert rows[0]['iteration-time'] == 2.0
    assert rows[0]['throughput'] == 0.8
    assert rows[0]['throughput/tokens_per_sec'] == 128.0
    assert rows[0]['throughput/samples_per_sec'] == 1.0
    assert rows[0]['train/consumed_input_tokens'] == 198 * 128
    assert rows[0]['memory/max_allocated_mib'] == 49397.14
    assert 'memory/max_allocated_mib' not in rows[1]
    assert rows[1]['train/skipped_iterations'] == 1
    assert rows[1]['distill/kl_loss'] == 13
    assert teacher['peak_active'] == 16
    assert all('teacher/request_seconds' not in row for row in rows)


def test_mixed_ce_loss_cannot_be_recovered_as_kl():
    rows, _ = parse_metrics(LOG, {'seq_length': 128, 'rwkv_distill_ce_weight': 1})
    assert all('distill/kl_loss' not in row for row in rows)


def test_unrecoverable_or_out_of_order_logs_fail_explicitly():
    with pytest.raises(ValueError, match='nonempty'):
        parse_metrics('no training metrics', {'seq_length': 128})
    with pytest.raises(ValueError, match='strictly increasing'):
        parse_metrics(LOG + LOG, {'seq_length': 128})
    with pytest.raises(ValueError, match='single-GPU'):
        parse_metrics(LOG, {'seq_length': 128, 'model_parallel': {'tensor_model_parallel_size': 2}})
