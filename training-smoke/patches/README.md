# Review and recovery

Changes are saved on the local `2026-10-10` branches; they have not been pushed. The parent baseline, frozen upstream revisions and Megatron adaptation commit are recorded in `../environment/provenance.json`. Repository URLs and author identity are documented in `../COMMIT-STATE.md`.

`root-tracked.patch` captures tracked parent-repository changes, including submodule pointers. `megatron-adaptation.patch` captures local compatibility fixes on Megatron upstream `d113016bb2850c4b7d804f24ba72c055d7d4f861`. FLA upstream `79d12e8ec0ae72bc7f2189b4244d6532ce8c268c` has no additional local changes. These patches do not include untracked files.

After the 2026-10-10 follow-up, FLA points at upstream40850297 with no custom source changes. The root patch excludes training-smoke because its scripts/reports are already saved in the source archive; this prevents generated patches from recursively including earlier copies of themselves.

`current-source.tar.gz` preserves the complete changed RWKV Python source, tests, and lightweight training-smoke scripts/configuration/reports, including untracked files. It excludes logs, caches, checkpoints, frozen replay snapshots and native build trees. Keep these excluded artifacts in place: checkpoints and source/native-extension build trees are separately retained in `../checkpoints/` and `../vendor/`.

To recover from local Git commits, use the root `2026-10-10` branch and its recorded submodule commits. For archive recovery, select the recorded parent and upstream submodule baselines, apply the Megatron patch inside its submodule, and restore the RWKV files from the source archive. Review existing local changes before applying patches or extracting an archive. Launch scripts and configurations currently contain absolute paths to this machine's isolated workspace and environment. The refreshed patches are cumulative relative to the recorded baselines even after committing.

Converter sync: source commit `68c9991bf6e0bc787b7b6f298a81c311fed3da96` was applied as a file patch, preserving existing work. Its original four-file patch is `../reference/converter-68c9991.patch`; `converter-upstream-compatibility.patch` records additional upstream pipeline-layout validation. The refreshed source archive also contains `rwkv_conversion.py`, both root READMEs and the converter tests. The pre-sync archive is preserved in `../reference/source-before-converter-sync.tar.gz`.

Single-GPU audit: `single-gpu-fla-layout.patch` records the two pointwise FLA layout adapters and their call sites relative to the pre-audit working copy. Production LayerNorm and CE/L2Wrap objectives were not switched. New regression tests and independent oracle/diagnostic scripts are included in the source archive. Details: `../audit/SINGLE-GPU.md`. Earlier performance tables precede this fix.

Subsequent LayerNorm change: RWKV7 now selects TE>=1.9.0 for ln0/ln1/ln2/ln_out, otherwise consistently falls back to Apex. Optional imports and disabled-FP8 contexts allow the missing-TE local BF16 path. No old-checkpoint backend mode was added; GroupNorm and CE/L2Wrap objectives remain unchanged. The updated archive contains normalization.py, new regression tests and separate TE evidence; `../reference/source-before-te-layernorm.tar.gz` preserves the earlier source snapshot. See `../audit/TE-UNIFICATION.md`.
