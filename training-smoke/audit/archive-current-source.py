"""Refresh lightweight recovery artifacts; leave original inputs/checkpoints intact."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'training-smoke/patches'
provenance = json.loads((ROOT / 'training-smoke/environment/provenance.json').read_text())
for repo, target, baseline in ((ROOT, 'root-tracked.patch', provenance['root_commit']),
                     (ROOT / '3rdparty/Megatron-LM-0.11.0', 'megatron-adaptation.patch', provenance['megatron_upstream'])):
    command = ['git', 'diff', '--binary', baseline]
    if repo == ROOT:
        # Reports are archived separately; avoid recursively embedding generated patches.
        command.extend(['--', '.', ':(exclude)training-smoke'])
    (PATH / target).write_bytes(subprocess.check_output(command, cwd=repo))
for repo, target in ((ROOT, 'root-status.txt'),
                     (ROOT / '3rdparty/Megatron-LM-0.11.0', 'megatron-status.txt')):
    (PATH / target).write_bytes(subprocess.check_output(['git', 'status', '--short'], cwd=repo))
files = {ROOT / name for name in ('.gitattributes', '.gitmodules', '.gitignore', 'README.md', 'README_EN.md', 'pretrain_rwkv7.py', 'rwkv_conversion.py')}
for directory in ('rwkv_megatron', 'tests', 'cuda'):
    files.update(p for p in (ROOT / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
excluded = {'logs', 'cache', 'checkpoints', 'vendor', 'reference', 'reference-repos', 'patches', '__pycache__'}
for p in (ROOT / 'training-smoke').rglob('*'):
    if not p.is_file() or any(part in excluded for part in p.relative_to(ROOT / 'training-smoke').parts):
        continue
    if p.suffix in ('.pth', '.pyc'):
        continue
    files.add(p)
files.update(p for p in PATH.iterdir() if p.is_file() and p.name not in ('manifest.json', 'current-source.tar.gz'))
archive = PATH / 'current-source.tar.gz'
with tarfile.open(archive, 'w:gz') as output:
    for p in sorted(files):
        output.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
manifest[str(archive.relative_to(ROOT))] = hashlib.sha256(archive.read_bytes()).hexdigest()
(PATH / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Archived', len(files), 'files,', archive.stat().st_size, 'bytes')
with tarfile.open(archive) as output:
    for p in files:
        relative = str(p.relative_to(ROOT))
        assert hashlib.sha256(output.extractfile(relative).read()).hexdigest() == manifest[relative]
print('Archive hashes verified')
