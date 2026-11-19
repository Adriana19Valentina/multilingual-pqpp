import argparse
import json
import os

import numpy as np

import retrieval_data as rd

SYSTEMS = ["clip", "clip-distilled", "clip-contrastive",
           "xlmr-vitb32", "xlmr-vitb32-distilled", "xlmr-vitb32-contrastive",
           "blip2", "blip2-distilled", "blip2-contrastive"]
COLUMNS = ["P@10", "R@10", "R@25", "MRR", "AP@10", "AP@25", "nDCG@10", "nDCG@25",
           "HR@10", "HR@25"]
DISCOUNT = 1.0 / np.log2(np.arange(2, 27))

parser = argparse.ArgumentParser(
    description="Standard retrieval metrics from the top-k lists saved by "
                "retrieve_multilingual.py.")
parser.add_argument("--systems", nargs="+", default=SYSTEMS)
parser.add_argument("--languages", nargs="+", default=rd.languages.LANGUAGES)
parser.add_argument("--split", default="test", choices=rd.SPLITS)
parser.add_argument("--source", default="mscoco", choices=["mscoco", "drawbench", "all"])
args = parser.parse_args()

relevance = rd.load_relevance()
frame = rd.languages.load_split(args.split)
index = rd.reference_results("clip", args.split)["index"]
rows = (np.flatnonzero((frame["source"] == args.source).to_numpy())
        if args.source != "all" else np.arange(len(frame)))


def metrics(top25, reciprocal_rank):
    out = {k: [] for k in COLUMNS if k != "MRR"}
    for position in rows:
        relevant = list(relevance.get(
            (args.split, frame["source"].iloc[position], int(index.iloc[position])), ()))
        if not relevant:
            continue
        gain = np.isin(top25[position], relevant).astype(float)
        for cut in (10, 25):
            g = gain[:cut]
            hits = g.sum()
            if cut == 10:
                out["P@10"].append(hits / cut)
            out[f"R@{cut}"].append(hits / len(relevant))
            out[f"HR@{cut}"].append(float(hits > 0))
            precision_at = np.cumsum(g) / np.arange(1, cut + 1)
            out[f"AP@{cut}"].append(float((precision_at * g).sum() / min(len(relevant), cut)))
            ideal = float(DISCOUNT[:min(len(relevant), cut)].sum())
            out[f"nDCG@{cut}"].append(float((g * DISCOUNT[:cut]).sum()) / ideal)
    summary = {k: float(np.mean(v)) for k, v in out.items() if v}
    summary["MRR"] = float(np.mean(reciprocal_rank[rows]))
    summary["queries"] = len(out["R@25"])
    return summary


table = {}
header = f"{'system':<26}{'language':<20}" + "".join(f"{c:>9}" for c in COLUMNS)
print(header)
print("-" * len(header))
for system in args.systems:
    for language in args.languages:
        path = os.path.join(rd.languages.EMBED_DIR, f"retrieval_{system}_{language}.npz")
        if not os.path.exists(path):
            continue
        data = np.load(path)
        if f"{args.split}_top25" not in data:
            continue
        summary = metrics(data[f"{args.split}_top25"], data[f"{args.split}_reciprocal_rank"])
        table[f"{system}|{language}"] = summary
        print(f"{system:<26}{language:<20}" + "".join(f"{summary[c]:>9.4f}" for c in COLUMNS))

if not table:
    raise SystemExit("no retrieval_<system>_<language>.npz found; run retrieve_multilingual.py")
print(f"\n{next(iter(table.values()))['queries']} queries, {args.source}, {args.split} "
      f"split. MRR is computed over the top 2000; everything else over the top 25.")

os.makedirs(rd.RESULTS_DIR, exist_ok=True)
out = os.path.join(rd.RESULTS_DIR, f"retrieval_metrics_{args.split}_{args.source}.json")
with open(out, "w") as handle:
    json.dump({"split": args.split, "source": args.source, "metrics": table}, handle, indent=2)
print(f"wrote {os.path.relpath(out, rd.languages.REPO)}")
