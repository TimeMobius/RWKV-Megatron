"""Aggregate canonical parameter names, excluding TP duplicates."""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def aggregate(folder, step):
    result = {}
    records = [json.loads(p.read_text()) for p in sorted(folder.glob(f"stats-step{step}-rank*.json"))]
    assert len(records) in (4, 8), (folder, step, len(records))
    for record in records:
        for row in record["parameters"]:
            if row["included"]:
                result[row["name"]] = result.get(row["name"], 0) + row["squared_norm"]
    norms = [row["independently_recomputed_norm"] for row in records]
    assert max(norms) == min(norms)
    return result, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="tp4pp2")
    parser.add_argument("--actual", default="tp2pp4")
    parser.add_argument("--step", type=int, default=26)
    parser.add_argument("--output", default="norm-comparison.json")
    args = parser.parse_args()
    a, ar = aggregate(HERE / args.reference, args.step)
    b, br = aggregate(HERE / args.actual, args.step)
    assert a.keys() == b.keys()
    rows = [dict(name=name, reference_norm=math.sqrt(a[name]), actual_norm=math.sqrt(b[name]),
                 squared_norm_delta=b[name] - a[name]) for name in a]
    rows.sort(key=lambda row: abs(row["squared_norm_delta"]), reverse=True)
    layers = {}
    for row in rows:
        parts = row["name"].split(".")
        key = ".".join(parts[:3]) if row["name"].startswith("decoder.layers.") else parts[0]
        agg = layers.setdefault(key, dict(reference_squared_norm=0, actual_squared_norm=0))
        agg["reference_squared_norm"] += row["reference_norm"] ** 2
        agg["actual_squared_norm"] += row["actual_norm"] ** 2
    result = dict(
        step=args.step,
        reference=dict(folder=args.reference, independent_norm=ar[0]["independently_recomputed_norm"],
                       reported_norm=ar[0]["optimizer_reported_norm"]),
        actual=dict(folder=args.actual, independent_norm=br[0]["independently_recomputed_norm"],
                    reported_norm=br[0]["optimizer_reported_norm"]),
        first_stage_batches_equal=next(r for r in ar if r["pp"] == r["tp"] == 0)["batches"]
        == next(r for r in br if r["pp"] == r["tp"] == 0)["batches"],
        last_stage_batches_equal=next(r for r in ar if r["pp"] == max(s["pp"] for s in ar) and r["tp"] == 0)["batches"]
        == next(r for r in br if r["pp"] == max(s["pp"] for s in br) and r["tp"] == 0)["batches"],
        parameters=rows,
        layers=sorted([dict(name=name, **row, squared_norm_delta=row["actual_squared_norm"] - row["reference_squared_norm"])
                       for name, row in layers.items()], key=lambda row: abs(row["squared_norm_delta"]), reverse=True),
    )
    (HERE / args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**{k: v for k, v in result.items() if k not in ("parameters", "layers")},
                      "worst_parameters": rows[:12], "worst_layers": result["layers"][:8]}, indent=2))


if __name__ == "__main__":
    main()
