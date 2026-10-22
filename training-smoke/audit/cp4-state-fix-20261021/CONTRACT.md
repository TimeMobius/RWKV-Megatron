# Recurrent CP state transport contract

Baseline: main 0ca9d3a, FLA 261dbcf5, branch 2026-10-10. Existing CP+SP working changes are preserved. Prior operator thresholds and CP1 numerical oracle are frozen.

| Source / reachable cell | Before | After / disposition | Oracle and gate |
| --- | --- | --- | --- |
| RWKV WKV7 CP1 | ordinary chunk_rwkv7 | existing unchanged implementation | prior tests |
| RWKV CP2/CP4 contiguous, each sample independent, BF16, TP/SP/ordinary PP | affine FLA summary | opt-in dplr_state_mode=recurrent; receive FP32 initial state, execute native state kernel once, send final FP32 state; reverse for backward | same CP1 BF16 computation; aligned 64-token shard boundaries: existing output<0.1%, each input gradient<1%, additionally bitwise equality on real 24-layer/all-head replay |
| FLA DPLR no CP, or CP default dplr_state_mode=affine | existing route | existing unchanged implementation | existing DPLR and CP regressions |
| FLA DPLR recurrent mode contiguous varlen | new opt-in | zero states at sequence boundaries; only partial first/last sequences communicate | ordinary full-varlen DPLR output/all six gradients; aligned cuts exact, other cuts existing BF16 tolerance |
| recurrent + zigzag / missing group / invalid mode | no former support | explicit ValueError before kernels | rejection tests |
| recurrent + TileLang requested | new opt-in | verifier rejects TileLang and dispatch uses recurrent Triton | route rejection checked without requiring TileLang package |

No recurrence load/operand/accumulation/store precision changes: existing BF16 operands, FP32 internal state arithmetic, and BF16 output staging are retained. Cross-rank states/gradients are FP32. Only state kernel walks incur cross-rank serial dependence; intra-chunk preparation, output, and parameter gradient kernels remain local.

WKV still builds one context per CP-local sequence geometry and loops over batch samples. Use dataclasses.replace to avoid mutating FLA cached contexts. No new model parameters, checkpoint keys, CLI options, or VPP changes.

Validation: real CP4/8192 all 24 layers and 36 heads; CP1/CP4 complete parameter-gradient comparison; CP4+SP/16384; CP2+ordinaryPP2+SP/MB2; both FLA backward cache modes; CP1/non-CP and original default-affine regression. Measure controlled full training old/new with identical configs, and label recurrent critical-path cost explicitly.
