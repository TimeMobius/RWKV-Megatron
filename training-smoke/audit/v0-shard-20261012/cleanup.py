"""Delete only this audit's test checkpoints after their strict comparison passes."""
import argparse
import json
import shutil
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('name',choices=('tp2-sp-pp2','tp4-sp-pp2'));args=parser.parse_args()
    assert json.loads((OUT/f'{args.name}-resume-validation.json').read_text())['exact_continuation']
    paths=[ROOT/f'training-smoke/checkpoints/v0-shard-20261012-{args.name}-{phase}' for phase in ('resume','continuous')]
    manifest=[]
    for path in paths:
        assert path.parent==ROOT/'training-smoke/checkpoints' and path.name.startswith('v0-shard-20261012-')
        manifest.extend({'path':str(f),'bytes':f.stat().st_size} for f in path.rglob('*') if f.is_file())
    total=sum(r['bytes'] for r in manifest)
    (OUT/f'{args.name}-checkpoint-cleanup.json').write_text(json.dumps({'validated_exact':True,'bytes':total,'files':manifest},indent=2)+'\n')
    for path in paths:shutil.rmtree(path)
    print('Freed GiB',total/2**30)


if __name__=='__main__':main()
