"""Put distillation charts in one W&B section without renaming logged metrics."""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

import wandb


ENTITY = "mobius"
PROJECT = "rwkv-qwen-distill-2p7b"
CHART_TITLES = {
    'throughput': 'throughput (TFLOP/s/GPU)',
    'iteration-time': 'iteration-time (s/step)',
    'throughput/tokens_per_sec': 'throughput (tokens/s)',
    'throughput/samples_per_sec': 'throughput (samples/s)',
}
QUERY = '''query Workspace($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    allViews(viewType: "project-view") {
      edges { node { id name displayName type spec } }
    }
  }
}'''
MUTATION = '''mutation SaveWorkspace($id: ID, $entity: String, $project: String,
  $name: String, $displayName: String, $spec: String) {
  upsertView(input: {id: $id, entityName: $entity, projectName: $project,
    name: $name, displayName: $displayName, type: "project-view", spec: $spec}) {
    view { id name } inserted
  }
}'''


def update_chart_titles(original):
    """Only rename existing panels; preserve axes, layout and selected runs."""
    spec = deepcopy(original)
    metrics = []
    for section in spec['section']['panelBankConfig']['sections']:
        for panel in section.get('panels', []):
            config = panel.get('config', {})
            keys = config.get('metrics', [])
            if len(keys) == 1 and keys[0] in CHART_TITLES:
                config['chartTitle'] = CHART_TITLES[keys[0]]
                metrics.append(keys[0])
    return spec, metrics


def build_spec(original, available):
    spec = deepcopy(original)
    section = spec['section']
    bank = section['panelBankConfig']
    original_charts = next(s for s in bank['sections'] if s['name'] == 'Charts')
    charts = deepcopy(original_charts)
    charts.update(isOpen=True, isPanelsAuto=False, panels=[])
    charts.pop('defaultName', None)
    preferred = ['lm loss', 'lm loss validation', 'grad-norm', 'learning-rate',
                 'throughput/tokens_per_sec', 'throughput', 'iteration-time',
                 'throughput/samples_per_sec', 'distill/kl_loss', 'memory/allocated_mib',
                 'memory/max_allocated_mib', 'memory/reserved_mib', 'memory/max_reserved_mib']
    allowed = [k for k in available if not k.startswith(('metrics_backfill/', 'teacher/final_'))
               and k != 'throughput/train_interval_tokens_per_sec']
    metrics = [k for k in preferred if k in allowed] + sorted(set(allowed) - set(preferred))
    restored_prefixes = ('train/', 'memory/', 'distill/', 'throughput/')
    for index, metric in enumerate(metrics):
        restored = metric.startswith(restored_prefixes) or metric in ('throughput', 'iteration-time')
        axis = 'train/iteration' if restored else '_step'
        panel_id = 'rwkv-' + hashlib.sha256(metric.encode()).hexdigest()[:12]
        charts['panels'].append({
            '__id__': panel_id, 'isAuto': False, 'viewType': 'Run History Line Plot',
            'layout': {'x': (index % 3) * 8, 'y': (index // 3) * 6, 'w': 8, 'h': 6},
            'config': {'chartTitle': CHART_TITLES.get(metric, metric), 'xAxis': axis, 'metrics': [metric]},
        })
    # Retain the existing system and hidden-section settings. Only the metric
    # prefix sections are replaced; all run data and run-table filters survive.
    bank['sections'] = [charts] + [deepcopy(s) for s in bank['sections']
                                   if s['name'] in ('System', 'Hidden Panels')]
    bank['settings']['sortAlphabetically'] = False
    section.setdefault('settings', {})['shouldAutoGeneratePanels'] = False
    section.setdefault('workspaceSettings', {})['shouldAutoGeneratePanels'] = False
    return spec, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--titles-only', action='store_true',
                        help='Rename existing charts without rebuilding the workspace')
    args = parser.parse_args()
    api = wandb.Api(overrides={'base_url': 'http://dian.sea-group.org:9098'})
    variables = {'entity': ENTITY, 'project': PROJECT}
    execute = api._service_api.execute_graphql
    edges = execute(QUERY, variables=variables)['project']['allViews']['edges']
    targets = [e['node'] for e in edges if e['node']['name'] == 'nw-nwusermobius-w']
    if len(targets) != 1:
        raise ValueError('Expected the existing Mobius project workspace')
    view = targets[0]
    original = json.loads(view['spec'])
    if args.titles_only:
        spec, metrics = update_chart_titles(original)
    else:
        available = set()
        for run in api.runs(f'{ENTITY}/{PROJECT}'):
            available.update(k for k, value in run.summary.items()
                             if not k.startswith('_') and isinstance(value, (int, float)))
        spec, metrics = build_spec(original, available)
    root = Path(__file__).resolve().parents[1]
    folder = root / 'training-smoke/qwen-rwkv-qwenvocab/workspace'
    folder.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    (folder / f'before-{stamp}.json').write_text(json.dumps(view, indent=2) + '\n')
    (folder / f'proposed-{stamp}.json').write_text(json.dumps(spec, indent=2) + '\n')
    assert original['section']['runSets'] == spec['section']['runSets']
    if not args.titles_only:
        assert len(metrics) == len(set(metrics))
        assert not any(s['name'] in ('train', 'throughput', 'memory', 'benchmark', 'validation', 'precision', 'distill', 'teacher')
                       for s in spec['section']['panelBankConfig']['sections'])
    if args.apply:
        # Re-read immediately before updating to avoid overwriting a user edit.
        current = execute(QUERY, variables=variables)['project']['allViews']['edges']
        current = next(e['node'] for e in current if e['node']['id'] == view['id'])
        if current['spec'] != view['spec']:
            raise RuntimeError('Workspace changed during preparation; rerun against the new settings')
        result = execute(MUTATION, variables={**variables, 'id': view['id'], 'name': view['name'],
                                             'displayName': view['displayName'], 'spec': json.dumps(spec)})
        assert not result['upsertView']['inserted']
        saved = execute(QUERY, variables=variables)['project']['allViews']['edges']
        saved = next(e['node'] for e in saved if e['node']['id'] == view['id'])
        actual = json.loads(saved['spec'])
        assert actual == spec
        (folder / 'latest.json').write_text(json.dumps(saved, indent=2) + '\n')
    print(json.dumps({'applied': args.apply, 'view': view['displayName'], 'charts': len(metrics),
                      'sections': [(s['name'], len(s['panels'])) for s in spec['section']['panelBankConfig']['sections']],
                      'url': f'http://dian.sea-group.org:9098/{ENTITY}/{PROJECT}',
                      'metrics': metrics}, indent=2))


if __name__ == '__main__':
    main()
