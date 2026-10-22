"""Independent FP64 local derivatives for the three sensitive first-token heads."""
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
torch.set_num_threads(8)
sources = {}
rows, fixtures = [], []
for folder, tp in [("ablate-tp4pp2-common-tp2", 4), ("ablate-tp2pp2-common-tp2", 2)]:
    for layer, head in [(15, 62), (19, 12), (23, 48)]:
        rank = head // (80 // tp)  # Selected layers all belong to PP2 stage0.
        local = head % (80 // tp)
        section = slice(local * 64, (local + 1) * 64)
        source_rank = (layer // 22) * 2 + head // 40
        if source_rank not in sources:
            sources[source_rank] = torch.load(HERE / "tp2pp4" / f"weights26-rank{source_rank}.pt",
                                              weights_only=True, mmap=True)["parameters"]
        weights = sources[source_rank]
        rho = weights[f"decoder.layers.{layer}.self_attention.rho"].reshape(-1, 64)[head % 40].double()
        gamma = weights[f"decoder.layers.{layer}.self_attention.group_norm.weight"].reshape(-1, 64)[head % 40].double()
        prefix = HERE / folder / f"layer{layer}-wkv-mb0-rank{rank}"
        data = torch.load(str(prefix) + ".pt", weights_only=True, mmap=True)
        r, k, v, decay, a, b = [t[0, 0, section].double().clone() for t in data["inputs"]]
        x = data["output"][0, 0, section].double().clone()
        do = torch.load(str(prefix) + "-grad.pt", weights_only=True)[0, 0, section].double()
        dy = torch.load(HERE / folder / f"layer{layer}-groupnorm-mb0-rank{rank}-grad.pt",
                        weights_only=True)[0, section].double()
        dr = torch.load(HERE / folder / f"layer{layer}-receptance-mb0-rank{rank}-grad.pt",
                        weights_only=True)[0, 0, section].double()
        # GroupNorm derivative on one 64-channel head, using the captured
        # BF16 input and incoming derivative, but independent FP64 arithmetic.
        centered = x - x.mean()
        variance = centered.square().mean()
        rstd = (variance + 0.00064).rsqrt()
        z = gamma * dy
        expected_do = (z - z.mean() - centered * (z * centered).mean() / (variance + 0.00064)) * rstd
        # At t=0, S_previous=0: WKV o=(r dot k)*v. Include the separate
        # RWKV output correction ut=(r*k*rho).sum()*v outside WKV.
        expected_dr = k * (do @ v) + k * rho * (dy @ v)

        def relative(actual, expected):
            return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))

        row = dict(folder=folder, layer=layer, head=head,
                   groupnorm_backward_vs_fp64_relative_l2=relative(do, expected_do),
                   receptance_backward_vs_fp64_relative_l2=relative(dr, expected_dr),
                   actual_receptance_gradient_norm=float(dr.norm()),
                   fp64_formula_gradient_norm=float(expected_dr.norm()),
                   groupnorm_rstd=float(rstd), groupnorm_incoming_gradient_norm=float(dy.norm()))
        rows.append(row)
        fixtures.append(dict(row=row, r=r, k=k, v=v, rho=rho, gamma=gamma, x=x, do=do, dy=dy, dr=dr))
(HERE / "token0-formula-check.json").write_text(json.dumps(rows, indent=2) + "\n")
torch.save(fixtures, HERE / "token0-formula-fixtures.pt")
print(json.dumps(rows, indent=2))
