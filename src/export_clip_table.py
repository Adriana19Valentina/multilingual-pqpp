import argparse
import json
import os
from datetime import datetime

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import languages

PIVOT = languages.PIVOT
TARGET = languages.TARGET_LANGUAGE
RAW_TARGET = languages.RAW_TARGET_LANGUAGE

HERE = languages.REPO
RESULTS_DIR = languages.RESULTS_DIR

COLUMNS = [
    ("glide", "GLIDE", "HBPP", "generation"),
    ("sdxl", "SDXL", "HBPP", "generation"),
    ("clip_p10", "CLIP", "P@10", 'retrieval'),
    ("clip_rr", "CLIP", "RR", 'retrieval'),
    ("blip2_p10", "BLIP-2", "P@10", 'retrieval'),
    ("blip2_rr", "BLIP-2", "RR", 'retrieval'),
]

PAPER = {
    "glide": (0.649, 0.474), "sdxl": (0.380, 0.246),
    "clip_p10": (0.473, 0.299), "clip_rr": (0.200, 0.149),
    "blip2_p10": (0.498, 0.358), "blip2_rr": (0.166, 0.150),
}

ROWS = [row for row in [
    ("control", PIVOT, PIVOT, "Control (Long-CLIP), pivot"),
    ("xlmr", PIVOT, PIVOT, "XLM-R, pivot"),
    ("xlmr", PIVOT, RAW_TARGET, "  pivot -> target (raw MT)") if RAW_TARGET else None,
    ("xlmr", PIVOT, TARGET, "  pivot -> target"),
    ("xlmr", TARGET, TARGET, "XLM-R, target language"),
    ("xlmr", TARGET, PIVOT, "  target -> pivot"),
] if row]

parser = argparse.ArgumentParser()
parser.add_argument("--decimal", default="comma", choices=["comma", "dot"])
parser.add_argument(
    "--decimals", type=int, default=5,
    help='how many decimals to print; default 5, because at 3 the difference between raw and reviewed translation vanishes through rounding (0.66014 and 0.65978 both become 0.660) although it is real',
)
cli = parser.parse_args()

def load(*parts):
    path = os.path.join(RESULTS_DIR, *parts)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)

def cell(variant, train_language, test_language, target):
    generative = target in ("glide", "sdxl")
    if generative:
        directory = "clip" if variant == "xlmr" else "clip_longclip-b"
        run = load(directory, f"{target}__{train_language}.json")
        node = (run or {}).get("evaluations", {}).get(test_language)
    else:
        directory = "clip_retrieval" if variant == "xlmr" else "clip_retrieval_longclip-b"
        run = load(directory, f"retrieval__{train_language}.json")
        evaluations = (run or {}).get("evaluations", {})

        node = evaluations.get(test_language, {}).get(target) or (
            evaluations.get(target) if test_language == PIVOT else None
        )
    return node.get("total") if node else None

def mark(p):
    return "‡" if p < 0.001 else ("†" if p < 0.01 else " ")

def number(value, decimals=None):
    if value is None:
        return ""
    text = f"{value:.{cli.decimals if decimals is None else decimals}f}"
    return text.replace(".", ",") if cli.decimal == "comma" else text

lines = []
add = lines.append
add("=" * 118)
add('Fine-tuned CLIP -- post-generation / post-retrieval predictor')
add(f"generated: {datetime.now():%Y-%m-%d %H:%M}")
add("=" * 118)
add("")
add('CLIP is frozen; only the 1024-512-256-1 MLP head is trained, over')
add('[text(512) ; image(512)]. Training on a translation therefore means training')
add('the head on translated embeddings, not adapting the encoder to that language.')
add("")
add('Generation: 4 images per prompt, predictions averaged. Retrieval: 25+25')
add('images per query, binary classification, then aggregation into P@10 / RR.')
add('The retrieval lists are reconstructed (the originals were never published).')
add("")
add('‡ p < 0.001   † p < 0.01   -- against the random baseline')
add("")
WIDTH = max(9, cli.decimals + 5)
add(" " * 32 + "".join(f"{m + ' ' + ms:>{2 * WIDTH}}" for _, m, ms, _ in COLUMNS))
add(" " * 32 + "".join(f"{'Pearson':>{WIDTH}}{'Kendall':>{WIDTH}}" for _ in COLUMNS))
add(" " * 32 + "".join(f"{task:>{2 * WIDTH}}" for _, _, _, task in COLUMNS))
add("-" * (32 + 2 * WIDTH * len(COLUMNS)))

paper_row = f"{'PQPP paper (Long-CLIP)':<32}"
for target, _, _, _ in COLUMNS:
    pearson, kendall = PAPER[target]
    paper_row += (f"{number(pearson, 3):>{WIDTH - 1}}‡{number(kendall, 3):>{WIDTH - 1}}‡")
add(paper_row)
add("  (the paper reports 3 decimals; the rest of the table has "
    f"{cli.decimals})")
add("-" * (32 + 2 * WIDTH * len(COLUMNS)))

table = []
for variant, train_language, test_language, label in ROWS:
    row = f"{label:<32}"
    values = {}
    for target, _, _, _ in COLUMNS:
        metrics = cell(variant, train_language, test_language, target)
        values[target] = metrics
        if metrics is None:
            row += f"{'--':>{WIDTH}}{'--':>{WIDTH}}"
        else:
            row += (f"{number(metrics['pearson']):>{WIDTH - 1}}{mark(metrics['pearson_p'])}"
                    f"{number(metrics['kendall']):>{WIDTH - 1}}{mark(metrics['kendall_p'])}")
    add(row)
    table.append((variant, train_language, test_language, label.strip(), values))

    if variant == "control":
        delta = f"{'  ^ delta control vs paper':<32}"
        for target, _, _, _ in COLUMNS:
            metrics = values[target]
            if metrics is None:
                delta += f"{'--':>{WIDTH}}{'--':>{WIDTH}}"
            else:
                delta += (f"{number(metrics['pearson'] - PAPER[target][0]):>{WIDTH}}"
                          f"{number(metrics['kendall'] - PAPER[target][1]):>{WIDTH}}")
        add(delta)
        add("")

add("")
add('Degradation under translation (pivot in-language -> pivot applied to the translation):')
add(f"  {'target':<12}{'piv->piv':<12}{'piv->tgt':<12}{'kept':<10}")
for target, _, _, _ in COLUMNS:
    source = cell("xlmr", PIVOT, PIVOT, target)
    transfer = cell("xlmr", PIVOT, TARGET, target)
    if not source or not transfer or source["pearson"] <= 0:
        add(f"  {target:<12}--")
        continue
    add(f"  {target:<12}{number(source['pearson']):<12}{number(transfer['pearson']):<12}"
        f"{transfer['pearson'] / source['pearson']:<10.1%}")

add("")
add('In-language training vs zero-shot transfer, both tested on the target language:')
add(f"  {'target':<12}{'piv->tgt':<12}{'tgt->tgt':<12}{'delta':<12}")
for target, _, _, _ in COLUMNS:
    transfer = cell("xlmr", PIVOT, TARGET, target)
    in_language = cell("xlmr", TARGET, TARGET, target)
    if not transfer or not in_language:
        add(f"  {target:<12}--")
        continue
    difference = in_language["pearson"] - transfer["pearson"]
    add(f"  {target:<12}{number(transfer['pearson']):<12}"
        f"{number(in_language['pearson']):<12}{number(difference):<12}")

text_path = os.path.join(RESULTS_DIR, "finetuned_clip_table.txt")
with open(text_path, "w") as handle:
    handle.write("\n".join(lines) + "\n")

header = ["Rand"] + [f"{m} {ms} {s}" for _, m, ms, _ in COLUMNS for s in ("Pearson", "Kendall")]
tsv = [header, ["PQPP paper (Long-CLIP)"] + [
    number(v) for target, _, _, _ in COLUMNS for v in PAPER[target]
]]
for _, _, _, label, values in table:
    tsv.append([label] + [
        number(values[target][statistic]) if values[target] else ""
        for target, _, _, _ in COLUMNS for statistic in ("pearson", "kendall")
    ])
tsv_path = os.path.join(RESULTS_DIR, "finetuned_clip_table.tsv")
with open(tsv_path, "w") as handle:
    handle.write("\n".join("\t".join(row) for row in tsv) + "\n")

json_path = os.path.join(RESULTS_DIR, "finetuned_clip_table.json")
with open(json_path, "w") as handle:
    json.dump({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "predictor": 'fine-tuned CLIP (CLIP frozen, the MLP head is trained)',
        "paper_baseline": {"encoder": "Long-CLIP", "values": {
            t: {"pearson": p, "kendall": k} for t, (p, k) in PAPER.items()}},
        "rows": [{"variant": v, "train_language": tr, "test_language": te,
                  "label": lb, "cells": cs} for v, tr, te, lb, cs in table],
    }, handle, indent=2, ensure_ascii=False)

print(f"wrote: {os.path.relpath(text_path, HERE)}")
print(f"wrote: {os.path.relpath(tsv_path, HERE)}")
print(f"wrote: {os.path.relpath(json_path, HERE)}")
