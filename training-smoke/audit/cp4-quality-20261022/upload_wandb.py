"""Publish the paired evaluation alongside its source training run."""

import json
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
PROJECT = 'mobius/Mobius-r7-28b-qwen-pretrain'


def main():
    summary = json.loads((HERE / 'summary.json').read_text())
    details = {name: json.loads((HERE / f'{name}-results.json').read_text())
               for name in ['initial', 'final']}
    run = wandb.init(
        entity='mobius', project='Mobius-r7-28b-qwen-pretrain',
        id='cp4-quality-20261022', resume='allow', mode='online',
        name='rwkv7-2p7b-cp4-halfday-quality-20261022', dir=str(HERE),
        job_type='evaluation', tags=['cp4', 'quality', 'paired-evaluation'],
        config=dict(source_train_run=f'{PROJECT}/runs/cp4-halfday-20261021',
                    checkpoint_iteration=49189, benchmark_n_per_task=100,
                    heldout_documents=256, heldout_length=512,
                    expanded_validation_sequences=256, expanded_validation_length=16384,
                    custom_likelihood_scorer=True, official_full_harness=False,
                    decoding='greedy, 128 new tokens, no penalties',
                    samples_sha256=details['final']['benchmark_samples_sha256']),
        settings=wandb.Settings(disable_git=True, x_disable_stats=True),
    )
    table = wandb.Table(columns=['task', 'metric', 'initial', 'final', 'n',
                                 'paired_change_ci_low', 'paired_change_ci_high'])
    for task in details['initial']['benchmarks']:
        metric = 'acc' if task in ['winogrande', 'lambada_openai'] else 'acc_norm'
        row = [task, metric]
        row += [details[name]['benchmarks'][task]['metrics'][metric] for name in ['initial', 'final']]
        row += [100, *summary['changes'][task]['paired_bootstrap_95pct_ci']]
        table.add_data(*row)
    generations = wandb.Table(columns=['prompt_name', 'prompt', 'initial_output', 'final_output'])
    for i, sample in enumerate(details['initial']['generations']):
        generations.add_data(sample['name'], sample['prompt'], sample['generated'],
                             details['final']['generations'][i]['generated'])
    for name, step in [('initial', 0), ('final', 49189)]:
        record = {'checkpoint_iteration': step}
        for key in ['ce', 'ppl', 'token_accuracy']:
            record[f'heldout/{key}'] = summary['models'][name]['heldout'][key]
        for key in ['loss', 'ppl']:
            record[f'expanded16k/{key}'] = summary['expanded'][name][key]
        for task, result in details[name]['benchmarks'].items():
            for key, value in result['metrics'].items():
                record[f'benchmarks/{task}/{key}'] = value
        run.log(record, step=step)
    run.log({'benchmark_comparison': table, 'generations': generations})
    run.summary.update(summary['models']['final']['heldout'])
    run.save(str(HERE / 'RESULTS.md'), base_path=str(HERE), policy='now')
    run.save(str(HERE / 'summary.json'), base_path=str(HERE), policy='now')
    url = run.url
    run.finish()
    api = wandb.Api(timeout=30)
    training = api.run(f'{PROJECT}/cp4-halfday-20261021')
    training.summary.update({
        'quality_evaluation_url': url,
        'quality_heldout_ppl_initial': summary['models']['initial']['heldout']['ppl'],
        'quality_heldout_ppl_final': summary['models']['final']['heldout']['ppl'],
        'quality_heldout_token_accuracy_final': summary['models']['final']['heldout']['token_accuracy'],
        'quality_expanded_16k_ppl_final': summary['expanded']['final']['ppl'],
    })
    (HERE / 'wandb-upload.json').write_text(json.dumps({'url': url}, indent=2) + '\n')
    print(url)


if __name__ == '__main__':
    main()
