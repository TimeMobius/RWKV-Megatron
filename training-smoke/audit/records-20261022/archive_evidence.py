"""Archive local evidence and dirty source without staging, committing, or deleting."""
import hashlib
import json
import os
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NOW = datetime.now(ZoneInfo("Asia/Shanghai"))
OUT = HERE / NOW.strftime("snapshot-%Y%m%d-%H%M%S")
OUT.mkdir(exist_ok=False)
REPOS = {"main": ROOT, "megatron": ROOT / "3rdparty/Megatron-LM-0.11.0",
         "fla": ROOT / "3rdparty/flash-linear-attention-260115"}
EVIDENCE = ["28b-oom-20261022", "tp-pp-gradnorm-20261022", "cp-sp-stage2-20261021",
            "cp-validation-20261021", "cp4-state-fix-20261021", "cp4-longrun-20261021",
            "cp4-quality-20261022", "cp-equivalence-20261021", "cp-error-source-20261021",
            "cp4-error-source-20261021", "cp-state-fix-20261021", "cp-stage1-20261021",
            "branch-isolation-20261022", "commit-20261022"]
TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".md", ".json", ".jsonl", ".csv",
                 ".log", ".txt", ".patch", ".diff"}
EXCLUDE_DIRS = {"__pycache__", ".git", "data-cache", "tensorboard", "wandb", "cache"}
MAX_TEXT = 128 * 2**20


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


state = {"captured_at_asia_shanghai": NOW.isoformat(), "root": str(ROOT),
         "remote_refs_fetched": False, "commits_created": False, "repositories": {}}
source_files = set()
for name, repo in REPOS.items():
    upstream = subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "@{upstream}"],
                              capture_output=True, text=True)
    tracking = upstream.stdout.strip() if upstream.returncode == 0 else None
    (OUT / f"{name}-worktree.patch").write_bytes(git(repo, "diff", "HEAD", "--binary", "--submodule=short"))
    (OUT / f"{name}-index.patch").write_bytes(git(repo, "diff", "--cached", "--binary", "--submodule=short"))
    changed = [x.decode() for x in git(repo, "diff", "HEAD", "--name-only", "-z").split(b"\0") if x]
    untracked = [x.decode() for x in git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if x]
    state["repositories"][name] = dict(
        path=str(repo), branch=git(repo, "branch", "--show-current").decode().strip(),
        head=git(repo, "rev-parse", "HEAD").decode().strip(), upstream=tracking,
        ahead_behind=(git(repo, "rev-list", "--left-right", "--count", f"HEAD...{tracking}").decode().strip()
                      if tracking else None),
        status_porcelain_v1=git(repo, "status", "--porcelain=v1", "--untracked-files=all").decode(),
        changed_tracked_files=changed, untracked_files=untracked)
    for rel in changed + untracked:
        path = repo / rel
        # Save dirty code/config/docs, including submodule untracked tests.
        if path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_TEXT:
            if path.suffix in TEXT_SUFFIXES or path.name == ".gitignore":
                # Audit files are selected below to exclude caches/large generated trees.
                if repo != ROOT or not rel.startswith("training-smoke/") or "/audit/" not in rel:
                    if not rel.startswith("training-smoke/wandb/"):
                        source_files.add(path)
state["submodule_status"] = git(ROOT, "submodule", "status", "--recursive").decode()
write_json(OUT / "repository-state.json", state)

paths = set(source_files)
for name in EVIDENCE:
    for directory, dirs, files in os.walk(ROOT / "training-smoke/audit" / name):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS)
        for file in files:
            path = Path(directory) / file
            if not path.is_symlink() and path.is_file() and path.suffix not in {".pyc", ".pyo"}:
                paths.add(path)
paths.update([ROOT / "TESTING_STATUS.md", ROOT / "todolist.txt", HERE / "README.md", Path(__file__).resolve()])
paths.update([ROOT / "BRANCH_WORKSPACES.md", ROOT / "COMMIT_AND_PUSH.md", ROOT / "push-recorded-branches.sh"])
inventory, archived = [], set()
for path in sorted(paths):
    size = path.stat().st_size
    eligible = (path.suffix in TEXT_SUFFIXES or path.name == ".gitignore") and size <= MAX_TEXT
    row = dict(path=str(path.relative_to(ROOT)), size_bytes=size, mtime_ns=path.stat().st_mtime_ns,
               archived=eligible, sha256=sha256(path) if eligible else None)
    if not eligible:
        row["not_archived_reason"] = "binary/tensor or exceeds 128MiB text limit; original retained"
    inventory.append(row)
    if eligible:
        archived.add(path)
write_json(OUT / "evidence-inventory.json", dict(root=str(ROOT), files=inventory,
           excluded_directory_names=sorted(EXCLUDE_DIRS), tensor_hashes_computed=False))
archive = OUT / "reviewable-evidence.tar.gz"
archived.update(OUT.glob("*.patch"))
archived.update([OUT / "repository-state.json", OUT / "evidence-inventory.json"])
with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
    for path in sorted(archived):
        tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
with tarfile.open(archive, "r:gz") as tar:
    members = tar.getmembers()
    assert len(members) == len(archived)
    assert all(m.isfile() and not Path(m.name).is_absolute() and ".." not in Path(m.name).parts for m in members)
summary = dict(snapshot_directory=str(OUT.relative_to(ROOT)), archive=str(archive.relative_to(ROOT)),
               archive_size_bytes=archive.stat().st_size, archive_sha256=sha256(archive),
               archived_file_count=len(archived), inventoried_file_count=len(inventory),
               local_only_retained_file_count=sum(not row["archived"] for row in inventory),
               local_only_retained_bytes=sum(row["size_bytes"] for row in inventory if not row["archived"]),
               captured_at_asia_shanghai=NOW.isoformat())
write_json(OUT / "backup-summary.json", summary)
(HERE / "LATEST.txt").write_text(str(OUT.relative_to(ROOT)) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
