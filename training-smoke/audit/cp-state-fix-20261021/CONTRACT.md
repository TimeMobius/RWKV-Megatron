# DPLR CP prefix-state fix contract

Baseline: FLA `ff481ea6862c99a818858081571373694de7b4e7`, parent branch `2026-10-10`.

This local bug fix aligns the DPLR CP prefix state's v2 operand staging with its ordinary DPLR state kernel. The baseline precision mismatch and real-production impact were established in `../cp-error-source-20261021/REPORT.md`. No public arguments, defaults, state formats, checkpoint formats, backend selection, or tolerances change. An upstream PR is outside this task; FLA's RFC requirement must be followed before publishing a precision-changing PR.

| Source / reachable route | Effective staging and disposition | Oracle and acceptance |
| --- | --- | --- |
| `rwkv_megatron.WKV7` → `chunk_rwkv7` → Triton DPLR CP, contiguous CP2, head64, safe_gate=True, BT64, BF16 | Corrected: w@BF16(h)+u accumulates FP32; v2 remains FP32 for FP32(bg) dot; state stored and communicated FP32 | Frozen CP1 production capture, full/uneven mask, exact captured forward; original gradient tolerances unchanged |
| Direct DPLR CP and RWKV7 wrapper, contiguous/zigzag, CP2/4, dense/varlen | Same corrected USE_BG leaf, all K slices through 256 | Existing CP naive-recurrence tests, output + all six gradients, existing ratio 8e-3 |
| DPLR prefix leaf, BF16 K64/128/192/256 V64, FP16 K64 V32/96, dense/varlen, BT64 | Corrected USE_BG leaf; FP32 state, unchanged FP32 decay and accumulation | Ordinary `chunk_dplr_fwd_h` final state, frozen ratio 5e-5 and finite checks; baseline head64 test must fail |
| DPLR TileLang CP caller | Shared prefix state receives same fix; public routing and later TileLang stages unchanged | Existing CP route test if optional backend is available; otherwise explicitly skipped |
| GDN, GDN2, KDA, gated delta product, intracard CP | Existing USE_BG=False route; original input-dtype v_new cast and dots retained | Caller audit plus representative existing CP regressions, original tolerances |

Addressing, grid, h/m layouts, gate bounds, head grouping and the affine-matrix branch are unchanged. USE_BG is a compile-time flag, so GDN/KDA compile without the new FP32 dot. Later CP4 affine-chain rounding remains subject to existing BF16 tolerances; the patch does not promise bitwise equality for every shape or CP size. The first regression reproducer failed before the fix at relative error 0.001067 against its frozen 5e-5 threshold.

No exponent regime or normalization predicate changes. Main-kernel FP32 dot precision follows the existing Triton environment; tests/benchmarks record the same environment for baseline and candidate. The production-representative before/after timing uses identical captured WKV inputs and same hardware; it is a local correctness fix, with no throughput-gain claim.
