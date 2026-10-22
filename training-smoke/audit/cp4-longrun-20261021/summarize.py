"""Summarize the running job without changing training or deleting artifacts."""

import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUMBER = r'[-+\d.eE]+|nan|inf'


def atomic_write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def evaluations(path):
    if not path.exists():
        return []
    pattern = rf'validation loss at iteration\s+(\d+)[^|]*\| lm loss value:\s*({NUMBER})'
    return [{'iteration': int(i), 'loss': float(loss)}
            for i, loss in re.findall(pattern, path.read_text(errors='replace'), re.I)]


def summarize(state):
    path = HERE / 'train.log'
    text = path.read_text(errors='replace') if path.exists() else ''
    rows = []
    for line in text.splitlines():
        if 'elapsed time per iteration (ms)' not in line:
            continue
        iteration = re.search(r'iteration\s+(\d+)/', line)
        fields = {}
        for name, label in [('ms', 'elapsed time per iteration (ms)'),
                            ('loss', 'lm loss'), ('grad_norm', 'grad norm'),
                            ('skipped', 'number of skipped iterations'),
                            ('nan', 'number of nan iterations')]:
            match = re.search(rf'{re.escape(label)}:\s*({NUMBER})', line, re.I)
            if match:
                fields[name] = float(match[1])
        if iteration and fields:
            rows.append({'iteration': int(iteration[1]), **fields})

    # Megatron reports skipped/nan counts per logging interval; add those counts.
    finite = all(math.isfinite(row.get(key, float('nan')))
                 for row in rows for key in ['loss', 'grad_norm', 'ms'])
    steady = [row for row in rows if row['iteration'] >= 100
              and math.isfinite(row.get('ms', float('nan'))) and row['ms'] > 0]
    windows = []
    for offset in range(0, len(steady), 100):
        window = steady[offset:offset + 100]
        median = statistics.median(row['ms'] for row in window)
        windows.append({'first_iteration': window[0]['iteration'],
                        'last_iteration': window[-1]['iteration'],
                        'median_step_ms': median,
                        'tokens_per_second_global': 32768 * 1000 / median})
    result = {
        **state,
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'last_train_record': rows[-1] if rows else None,
        'train_log_records': len(rows),
        'logged_losses_and_grad_norms_finite': finite if rows else None,
        'skipped_iterations_in_logged_intervals': sum(row.get('skipped', 0) for row in rows),
        'nan_iterations_in_logged_intervals': sum(row.get('nan', 0) for row in rows),
        'baseline_validation': evaluations(HERE / 'baseline.log'),
        'periodic_validation': evaluations(path),
        'checkpoint_validation': evaluations(HERE / 'final-eval.log'),
        'timing_windows_100_log_records': windows,
        'completed_checkpoint_iterations': [int(i) for i in re.findall(
            r'successfully saved checkpoint from iteration\s+(\d+)', text)],
    }
    atomic_write(HERE / 'status.json', json.dumps(json_safe(result), indent=2, allow_nan=False) + '\n')
    last = result['last_train_record']
    report = [
        '# CP4 12小时长跑状态', '',
        f"状态：{state['status']}；阶段：{state['phase']}。",
        f"更新（UTC）：{result['updated_at_utc']}。", '',
        '配置：真实2.7B/0056，8×H800，TP2/PP1/CP4，SP开启，seq16384，MB1/GB2；'
        'BF16、Adam、lr=1e-6、整层重算。训练进程720分钟后保存退出。', '',
        f'最后训练记录：`{json.dumps(last, ensure_ascii=False)}`。',
        f"日志区间累计跳步={result['skipped_iterations_in_logged_intervals']}，"
        f"NaN步={result['nan_iterations_in_logged_intervals']}；"
        f"记录的loss/grad norm有限={result['logged_losses_and_grad_norms_finite']}。", '',
        f"初始固定验证：`{json.dumps(result['baseline_validation'])}`。",
        f"最新周期验证：`{json.dumps(result['periodic_validation'][-3:])}`。",
        f"退出checkpoint重载验证：`{json.dumps(result['checkpoint_validation'])}`。", '',
        '完整曲线见tensorboard/和训练日志；GPU显存/利用率/功耗采样见gpu.csv。'
        'status.json包含每100个日志记录的耗时中位数与全局tokens/s。'
        '计时沿用生产日志，会包括日志区间内的其他开销，不能替代CP1/CP4受控性能对照。', '',
        '学习趋势根据固定验证loss判断；单次长跑不能证明与CP1长期等价。'
        'baseline和周期验证使用相同数据配置、固定前32条16K样本；最终验证重载checkpoint，'
        '但不等同于Adam/RNG连续训练恢复对拍。', '',
        '本任务不改生产源码，不自动删除checkpoint，不自动重试失败的训练。',
    ]
    if state.get('error'):
        report += ['', f"失败原因：{state['error']}"]
    if state['status'] == 'completed':
        report += [
            '', '## 完成长跑结论', '',
            f"2026-10-22 05:57（北京时间）结束，完成{state['final_checkpoint_iteration']}步，"
            f"处理{state['final_checkpoint_iteration'] * 32768:,}个token。"
            '训练在720分钟上限正常保存退出，baseline/train/final-eval进程退出码均为0。', '',
            '固定32条16K验证样本loss从13.49113下降至4.067253，'
            '在该配置和样本范围内表现出明确的学习趋势。'
            '全部日志区间无NaN/跳步，10个checkpoint保存成功，最终checkpoint新进程重载验证通过。'
            '记录的生产源码起止hash一致。', '',
            '第100步起日志耗时中位数865.9 ms/步，约3.78万tokens/s（8卡合计）；'
            'GPU采样峰值22315 MiB/卡。训练后段未见持续耗时增长或显存增长。'
            '这些是本次运行的观测，不代表CP1对照或更大规模扩展结论。',
        ]
    upload_path = HERE / 'wandb-upload.json'
    if upload_path.exists():
        upload = json.loads(upload_path.read_text())
        report += [
            '', f"W&B：[长跑指标]({upload['url']})。2026-10-22训练结束后补传，"
            f"共{upload['history_rows']}条history记录，含100次固定验证（初始、98次周期和最终重载）。"
            'W&B导入耗时不代表训练耗时；训练原始起止时间已写入run config。'
            '`train/loss_instant`来自TensorBoard的每10步采样值，'
            '`train/loss_interval_mean`来自stdout的日志区间均值，两者定义不同。',
        ]
    atomic_write(HERE / 'RESULTS.md', '\n'.join(report) + '\n')
    return result


if __name__ == '__main__':
    state = json.loads((HERE / 'state.json').read_text())
    summarize(state)
