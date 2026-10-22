"""Paired comparisons with document/question bootstrap intervals."""

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def comparison(a, b):
    change = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    resamples = np.random.default_rng(20261022).integers(len(a), size=(10000, len(a)))
    low, high = np.quantile(change[resamples].mean(-1), [.025, .975])
    return dict(mean_change=float(change.mean()), paired_bootstrap_95pct_ci=[float(low), float(high)])


def main():
    models = {name: json.loads((HERE / f'{name}-results.json').read_text()) for name in ['initial', 'final']}
    a, b = models['initial'], models['final']
    assert a['benchmark_samples_sha256'] == b['benchmark_samples_sha256']
    assert a['heldout_samples_sha256'] == b['heldout_samples_sha256']
    changes = {'heldout': {metric: comparison(
        [r[metric] for r in a['heldout']['per_sample']],
        [r[metric] for r in b['heldout']['per_sample']]) for metric in ['ce', 'token_accuracy']}}
    lines = [
        '# CP4半天训练后的模型效果', '',
        '**结论：训练改善了语料内文本预测，但常用基准没有一致提升，当前自由生成仍严重重复、答非所问。**'
        '10组固定提示词中，中文问题、简单算术、翻译、摘要和代码样例均未正确完成；'
        '英文续写能组成句子，但经常循环。'
        '此结论限定于本报告的抽样基准和贪心生成，不等于完整模型能力认证。', '',
        '初始裁剪2.7B权重与step49189做同题、同分词、同BF16生产Megatron/FLA前向对比。'
        '最终TP2权重通过仓库转换器合并为TP1推理权重；转换核验见export-check.json。', '',
        '## 验证文档预测', '',
        '0056 validation划分中重新固定抽取256个不同文档，每个513 tokens，预测512个位置。'
        '文档ID与训练划分不交叉，并排除长跑固定32条验证样本涉及的文档；'
        '未排除语料之间的语义重复。CE以有效248077词表归一化。', '',
        '| 模型 | CE ↓ | PPL ↓ | next-token准确率 ↑ |',
        '| --- | ---: | ---: | ---: |',
    ]
    for name, result in models.items():
        h = result['heldout']
        lines.append(f"| {name} | {h['ce']:.6f} | {h['ppl']:.3f} | {h['token_accuracy']:.2%} |")
    lines += ['', '配对文档bootstrap（10000次）的CE差值95%区间：'
              f"`{changes['heldout']['ce']}`。", '', '## 扩大16K验证', '']
    expanded = {}
    for name in models:
        text = (HERE / f'{name}-expanded.log').read_text()
        match = re.search(r'validation loss at iteration.*?lm loss value:\s*([\d.E+-]+).*?lm loss PPL:\s*([\d.E+-]+)', text)
        assert match, name
        expanded[name] = dict(loss=float(match[1]), ppl=float(match[2]))
    lines += ['两份模型使用相同seed、数据索引、256条16K序列（4194304预测token），'
              '沿用TP2/CP4/SP与生产loss，未做训练更新。该组扩展验证可能包含原固定验证中的部分内容。', '',
              '| 模型 | CE ↓ | PPL ↓ |', '| --- | ---: | ---: |']
    for name, row in expanded.items():
        lines.append(f"| {name} | {row['loss']:.6f} | {row['ppl']:.3f} |")
    lines += ['', '## 常用基准固定抽样', '',
              '每项100题，零样本，无chat模板，复用2026-10-21归档题目和lm-eval任务格式。'
              '这是自定义likelihood评分器，非官方harness完整榜单；'
              '公开基准是否出现在训练语料中未经排查。', '',
              '| 任务 | 指标 | 初始 | 最终 | 配对差值95%区间 |',
              '| --- | --- | ---: | ---: | --- |']
    for task in a['benchmarks']:
        metric = 'acc' if task in ['winogrande', 'lambada_openai'] else 'acc_norm'
        key = 'exact_match' if task == 'lambada_openai' else 'correct' if task == 'winogrande' else 'correct_norm'
        first = a['benchmarks'][task]
        last = b['benchmarks'][task]
        assert [r['source_index'] for r in first['per_sample']] == [r['source_index'] for r in last['per_sample']]
        delta = comparison([r[key] for r in first['per_sample']], [r[key] for r in last['per_sample']])
        changes[task] = delta
        lo, hi = delta['paired_bootstrap_95pct_ci']
        lines.append(f"| {task} | {metric} | {first['metrics'][metric]:.1%} | "
                     f"{last['metrics'][metric]:.1%} | [{lo:+.1%}, {hi:+.1%}] |")
    lines += ['', 'LAMBADA另有最后一个词的PPL，见逐模型JSON；与token PPL定义不同。'
              '每项100题存在明显抽样误差，数道题的变化不宜直接归因为通用能力提升。', '',
              '## 相同提示词的实际生成', '',
              '贪心解码，最多128个新token，不加重复惩罚。'
              '包含续写、代码、问答、算术、翻译、摘要和chat模板；'
              '本轮是预训练延续，没有额外指令微调。以下全部提示词与输出原样保留，未挑选最好样本。']
    for i, generation in enumerate(a['generations']):
        other = b['generations'][i]
        assert generation['prompt'] == other['prompt']
        lines += ['', f"### {generation['name']}", '', '提示词：', '```text', generation['prompt'], '```']
        for name, result in models.items():
            lines += ['', name + '：', '```text', result['generations'][i]['generated'], '```']
    checks = json.loads((HERE / 'export-check.json').read_text())
    lines += ['', '## 评测校验与复现', '',
              'TP2原checkpoint与导出后重新载入TP2的64个位置、完整248320词表logits逐位一致，'
              '确认该输入下转换往返正确。TP2原checkpoint与TP1评测的对应logits相对L2为'
              f"{checks['relative_l2']:.4%}，top-1一致率{checks['top1_agreement']:.2%}；"
              '跨TP布局不是逐位相同，所有成对质量评测统一使用TP1。', '',
              '两模型重复前向逐位一致、CE独立公式校验通过、logits有限；'
              '生成时选定位置head输出与完整logits对应位置逐位一致。'
              '初始模型六任务acc与既有归档评测一致。', '',
              '在仓库根目录，使用train_mobius_py312；'
              '`bash training-smoke/audit/cp4-quality-20261022/launch.sh final`复现最终模型评测，'
              '其他模式见launch.sh。summarize.py复核已有结果；不重新训练。'
              '权重导出和校验日志见export.log、export-check.log；原权重、checkpoint不修改。']
    summary = dict(models={name: {'heldout': {k: v for k, v in result['heldout'].items() if k != 'per_sample'},
                                  'benchmarks': {task: v['metrics'] for task, v in result['benchmarks'].items()},
                                  'checks': result['checks']} for name, result in models.items()},
                   expanded=expanded, changes=changes, export_check=checks)
    (HERE / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (HERE / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
