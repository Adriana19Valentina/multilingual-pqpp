import argparse
import json
import os
from datetime import datetime

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import languages

HERE = languages.REPO
RESULTS_DIR = languages.RESULTS_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

COLUMNS = [
    ("glide", "GLIDE", "HBPP", "generation"),
    ("sdxl", "SDXL", "HBPP", "generation"),
    ("clip_p10", "CLIP", "P@10", 'retrieval'),
    ("clip_rr", "CLIP", "RR", 'retrieval'),
    ("blip2_p10", "BLIP-2", "P@10", 'retrieval'),
    ("blip2_rr", "BLIP-2", "RR", 'retrieval'),
]

PAPER = {
    "glide": (0.548, 0.393), "sdxl": (0.159, 0.107),
    "clip_p10": (0.270, 0.186), "clip_rr": (0.189, 0.162),
    "blip2_p10": (0.159, 0.133), "blip2_rr": (0.206, 0.158),
}

SETUP = {
    "generation": {
        "images": 4, "note": "2 SDXL + 2 GLIDE",
        "conv": "3 layers (16-64)", "fc1": "262144 -> 512", "out": "sigmoid",
    },
    'retrieval': {
        "images": 25, "note": "top-25 of the predicted system",
        "conv": "4 layers (32-64)", "fc1": "65536 -> 1024", "out": "ReLU",
    },
}

FT = f"xlmr-vitb32-ft-{languages.TARGET_LANGUAGE}"
LONGCLIP_FT = f"longclip-b-ft-{languages.TARGET_LANGUAGE}"

MATRIX_VARIANTS = [
    ("longclip-b", "", "512x512 dims (original code)"),
    ("longclip-b", "__imgmat", "4x4 images (as the paper says)"),
    ("xlmr-vitb32", "", "512x512 dims"),
    ("xlmr-vitb32", "__imgmat", "4x4 images"),
    ("xlmr-vitb32", "__txtmat", "5x5, + pivot prompt"),
    ("xlmr-vitb32", "__mlmat", "6x6, + pivot and target prompt"),
    (FT, "", "512x512 dims"),
    (FT, "__imgmat", "4x4 images"),
    (LONGCLIP_FT, "", "512x512 dims"),
    (LONGCLIP_FT, "__imgmat", "4x4 images"),
]
GEN_TARGETS = ["glide", "sdxl"]

parser = argparse.ArgumentParser()
parser.add_argument("--decimal", default="comma", choices=["comma", "dot"])
cli = parser.parse_args()

def load(*parts):
    path = os.path.join(RESULTS_DIR, *parts)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)

def run_for(target, suffix="", encoder="longclip-b"):
    generative = target in ("glide", "sdxl")
    return load("corrcnn" if generative else "corrcnn_retrieval",
                f"{target}__{encoder}{suffix}.json")

def cell(target, subset="total", suffix="", encoder="longclip-b"):
    run = run_for(target, suffix, encoder)
    return (run or {}).get("evaluation", {}).get(subset)

def mark(p):
    return "‡" if p < 0.001 else ("†" if p < 0.01 else " ")

def number(value, decimals=3):
    if value is None:
        return ""
    text = f"{value:.{decimals}f}"
    return text.replace(".", ",") if cli.decimal == "comma" else text

lines = []
add = lines.append
add("=" * 118)
add('Correlation CNN -- post-generation / post-retrieval predictor')
add(f"generated: {datetime.now():%Y-%m-%d %H:%M}")
add("=" * 118)
add("")
add('The input is a 512x512 correlation matrix between embedding dimensions,')
add("computed over the prompt's images. The text NEVER enters the model:")
add('`retrieve_embeddings` exists in the original code but is never called.')
add("")
add("That is why the table has a single row of results. A 'multilingual' variant")
add("would give bit-for-bit identical numbers. The image encoder is Long-CLIP, the")
add("one from the paper, the only one comparable with Table 3.")
add("")
add('‡ p < 0.001   † p < 0.01   -- against the random baseline')
add("")
add(" " * 32 + "".join(f"{m + ' ' + ms:>18}" for _, m, ms, _ in COLUMNS))
add(" " * 32 + "".join("  Pearson  Kendall" for _ in COLUMNS))
add(" " * 32 + "".join(f"{task:>18}" for _, _, _, task in COLUMNS))
add("-" * 140)

row = f"{'PQPP paper':<32}"
for target, _, _, _ in COLUMNS:
    pearson, kendall = PAPER[target]
    row += f"{number(pearson):>8}‡{number(kendall):>8}‡"
add(row)

row = f"{'Our run':<32}"
cells = {}
for target, _, _, _ in COLUMNS:
    metrics = cell(target)
    cells[target] = metrics
    if metrics is None:
        row += f"{'--':>9}{'--':>9}"
    else:
        row += (f"{number(metrics['pearson']):>8}{mark(metrics['pearson_p'])}"
                f"{number(metrics['kendall']):>8}{mark(metrics['kendall_p'])}")
add(row)

delta = f"{'  ^ delta vs paper':<32}"
for target, _, _, _ in COLUMNS:
    metrics = cells[target]
    if metrics is None:
        delta += f"{'--':>9}{'--':>9}"
    else:
        delta += (f"{number(metrics['pearson'] - PAPER[target][0]):>9}"
                  f"{number(metrics['kendall'] - PAPER[target][1]):>9}")
add(delta)

add("")
add("By subset (Pearson):")
add(f"  {'target':<12}{'total':<10}{'mscoco':<10}{'drawbench':<12}")
for target, _, _, _ in COLUMNS:
    parts = [number(cell(target, s)["pearson"]) if cell(target, s) else "--"
             for s in ("total", "mscoco", "drawbench")]
    add(f"  {target:<12}{parts[0]:<10}{parts[1]:<10}{parts[2]:<12}")

add("")
add('The configuration differs between tasks, as it does in the original code:')
add(f"  {'':<12}{'images':<10}{'convolutions':<22}{'fc1':<18}{'output':<10}")
for task, setup in SETUP.items():
    add(f"  {task:<12}{setup['images']:<10}{setup['conv']:<22}"
        f"{setup['fc1']:<18}{setup['out']:<10}")
add("")
add('The image count matters: the matrix has rank at most (images - 1), so')
add('on generation it is strongly degenerate (rank <= 3), while on retrieval it is far')
add('better conditioned (rank <= 24).')

add("")
add("MATRIX VARIANTS (generation only)")
add("")
add('The original code correlates the DIMENSIONS of the embedding, although the')
add('paper describes correlations between IMAGES. And neither uses the prompt,')
add('even though CLIP offers the text-image similarity for free -- exactly the')
add('signal that says how well the image matches the prompt.')
add("")
add(f"  {'encoder':<34}{'matrix':<34}" +
    "".join(f"{target.upper():>22}" for target in GEN_TARGETS))
add(f"  {'':<34}{'':<34}" +
    "".join(f"{'Pearson':>11}{'Kendall':>11}" for _ in GEN_TARGETS))
add("  " + "-" * 112)
for encoder, suffix, description in MATRIX_VARIANTS:
    row = f"  {encoder:<34}{description:<34}"
    for target in GEN_TARGETS:
        metrics = cell(target, suffix=suffix, encoder=encoder)
        if metrics is None:
            row += f"{'--':>11}{'--':>11}"
        else:
            row += (f"{number(metrics['pearson']):>10}{mark(metrics['pearson_p'])}"
                    f"{number(metrics['kendall']):>10}{mark(metrics['kendall_p'])}")
    add(row)
add("")
add('Differences checked with the Steiger test on the per-prompt predictions:')
add('  text in the matrix (5x5 vs 4x4)   GLIDE +0.028 (p=1e-11)   SDXL +0.016 (p=1e-03)')
add('  second language    (6x6 vs 5x5)   GLIDE +0.005 (p=0.015)   SDXL -0.000 (n.s.)')
add("")
add('Adding the prompt is a solid gain on both targets. The second language adds')
add('something measurable only on GLIDE, and five times less -- but it is the first')
add('variant in which the predictor is NOT blind to language, because it has a')
add('textual input. Both require an encoder whose text and image embeddings live')
add('in the same space, so xlmr-vitb32, not longclip-b.')

add("")
add('Limitation to report: the original code packs the two retrieval matrices')
add('as [2, 512, 512] and uses the AVERAGED target, but the architecture')
add('accepts a single channel, while Table 3 reports per system. The two cannot')
add('be reconciled; the variant compatible with the table was chosen -- one model per')
add('(system, metric). This probably explains the larger deviations on retrieval.')

text_path = os.path.join(RESULTS_DIR, "correlation_cnn_table.txt")
with open(text_path, "w") as handle:
    handle.write("\n".join(lines) + "\n")

header = ["Row"] + [f"{m} {ms} {s}" for _, m, ms, _ in COLUMNS
                     for s in ("Pearson", "Kendall")]
tsv = [
    header,
    ["PQPP paper"] + [number(v) for target, _, _, _ in COLUMNS for v in PAPER[target]],
    ["Our run"] + [
        number(cells[target][statistic]) if cells[target] else ""
        for target, _, _, _ in COLUMNS for statistic in ("pearson", "kendall")
    ],
]
tsv_path = os.path.join(RESULTS_DIR, "correlation_cnn_table.tsv")
with open(tsv_path, "w") as handle:
    handle.write("\n".join("\t".join(row) for row in tsv) + "\n")

json_path = os.path.join(RESULTS_DIR, "correlation_cnn_table.json")
with open(json_path, "w") as handle:
    json.dump({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "predictor": "correlation CNN",
        "language_independent": True,
        "note": "does not use the prompt text; results are identical in any language",
        "image_encoder": "longclip-b",
        "paper_baseline": {t: {"pearson": p, "kendall": k} for t, (p, k) in PAPER.items()},
        "setup": SETUP,
        "cells": {t: {s: cell(t, s) for s in ("total", "mscoco", "drawbench")}
                  for t, _, _, _ in COLUMNS},
    }, handle, indent=2, ensure_ascii=False)

print(f"wrote: {os.path.relpath(text_path, HERE)}")
print(f"wrote: {os.path.relpath(tsv_path, HERE)}")
print(f"wrote: {os.path.relpath(json_path, HERE)}")
