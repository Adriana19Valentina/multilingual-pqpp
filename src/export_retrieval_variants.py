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
T = ["clip_p10", "clip_rr", "blip2_p10", "blip2_rr"]
LABELS = {"clip_p10": "CLIP P@10", "clip_rr": "CLIP RR",
          "blip2_p10": "BLIP-2 P@10", "blip2_rr": "BLIP-2 RR"}
PAPER = {"clip_p10": 0.473, "clip_rr": 0.200, "blip2_p10": 0.498, "blip2_rr": 0.166}

ROWS = [
    ("— original code and controls —", None, None, None),
    ("PQPP paper (Long-CLIP)", None, "PAPER", None),
    ("Long-CLIP control, aggregation a", "clip_retrieval_longclip-b", f"retrieval__{PIVOT}", PIVOT),
    ("Long-CLIP control, aggregation b", "clip_retrieval_longclip-b", f"retrieval__{PIVOT}__b", PIVOT),
    ('XLM-R, concat, aggregation a', "clip_retrieval", f"retrieval__{PIVOT}", PIVOT),
    ('XLM-R, concat, aggregation b', "clip_retrieval", f"retrieval__{PIVOT}__b", PIVOT),

    ("— interaction features —", None, None, None),
    ("interaction, piv → piv", "clip_retrieval", f"retrieval__{PIVOT}__b__inter", PIVOT),
    ("interaction, piv → tgt", "clip_retrieval", f"retrieval__{PIVOT}__b__inter", TARGET),
    ("interaction, tgt → tgt", "clip_retrieval", f"retrieval__{TARGET}__b__inter", TARGET),

    ('— trainable text tower —', None, None, None),
    ("interaction + text tower, piv → piv", "clip_retrieval", f"retrieval__{PIVOT}__b__inter__tt", PIVOT),
    ("interaction + text tower, piv → tgt", "clip_retrieval", f"retrieval__{PIVOT}__b__inter__tt", TARGET),
    ("interaction + text tower, tgt → tgt (seq.)", "clip_retrieval",
     f"retrieval__{TARGET}__b__inter__tt__seq", TARGET),

    ("— multilingual augmentation —", None, None, None),
    ("augmented, piv+tgt → piv", "clip_retrieval", f"retrieval__{PIVOT}+{TARGET}__b__inter", PIVOT),
    ("augmented, piv+tgt → tgt", "clip_retrieval", f"retrieval__{PIVOT}+{TARGET}__b__inter", TARGET),
    ("augmented, piv+both → piv", "clip_retrieval",
     f"retrieval__{PIVOT}+{RAW_TARGET}+{TARGET}__b__inter", PIVOT),
    ("augmented, piv+both → tgt", "clip_retrieval",
     f"retrieval__{PIVOT}+{RAW_TARGET}+{TARGET}__b__inter", TARGET),
    ("augmented, piv+tgt + consistency → tgt", "clip_retrieval",
     f"retrieval__{PIVOT}+{TARGET}__b__inter__cons0.5", TARGET),

    ("— sequential (continued from the pivot) —", None, None, None),
    ("sequential, tgt → tgt", "clip_retrieval", f"retrieval__{TARGET}__b__inter__seq", TARGET),
    ("sequential, + augm piv+tgt → tgt", "clip_retrieval",
     f"retrieval__{PIVOT}+{TARGET}__b__inter__seq", TARGET),
    ("sequential, + augm both → tgt", "clip_retrieval",
     f"retrieval__{PIVOT}+{RAW_TARGET}+{TARGET}__b__inter__seq", TARGET),
    ("sequential, + augm + text tower → piv", "clip_retrieval",
     f"retrieval__{PIVOT}+{TARGET}__b__inter__tt__seq", PIVOT),
    ("sequential, + augm + text tower → tgt", "clip_retrieval",
     f"retrieval__{PIVOT}+{TARGET}__b__inter__tt__seq", TARGET),
]

parser = argparse.ArgumentParser()
parser.add_argument("--decimal", default="comma", choices=["comma", "dot"])
cli = parser.parse_args()

def number(value):
    if value is None:
        return ""
    text = f"{value:.3f}"
    return text.replace(".", ",") if cli.decimal == "comma" else text

def cells(directory, name, language):
    if name == "PAPER":
        return {t: PAPER[t] for t in T}, None
    path = os.path.join(RESULTS_DIR, directory, f"{name}.json")
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as handle:
            run = json.load(handle)
    except json.JSONDecodeError:

        return None, "json trunchiat"
    evaluations = run["evaluations"]
    node = evaluations.get(language) or evaluations
    return {t: node[t]["total"]["pearson"] for t in T if t in node}, None

lines = []
add = lines.append
add("=" * 92)
add('Post-retrieval predictor: variants tested')
add(f"generated: {datetime.now():%Y-%m-%d %H:%M}")
add("=" * 92)
add("")
add('Encoder XLM-R ViT-B/32 (frozen unless stated otherwise). The retrieval')
add('retrieval lists are reconstructed and validated against published targets.')
add('pivot -> target means: trained on the pivot language, tested on the translation.')
add("")
add(f"{'variant':<38}" + "".join(f"{LABELS[t]:>13}" for t in T))
add("-" * 92)

table, missing = [], []
for label, directory, name, language in ROWS:
    if name is None:
        add("")
        add(label)
        continue
    values, problem = cells(directory, name, language)
    if values is None:
        add(f"  {label:<36}" + f"{'(' + (problem or 'missing') + ')':>13}")
        missing.append((label, problem or "missing"))
        continue
    add(f"  {label:<36}" + "".join(f"{number(values.get(t)):>13}" for t in T))
    table.append({"label": label, "run": name, "test_language": language,
                  "cells": values})

add("")
add("-" * 92)
add("Best variant per target:")
for t in T:
    best = max((row for row in table if t in row["cells"]),
               key=lambda row: row["cells"][t])
    add(f"  {LABELS[t]:<14}{number(best['cells'][t]):>8}   {best['label']}")

add("")
add('Gain over the original code (concat, aggregation a, pivot only):')
baseline = next((r for r in table if r["run"] == f"retrieval__{PIVOT}"
                 and r["test_language"] == PIVOT), None)
if baseline:
    best_overall = {t: max(r["cells"][t] for r in table if t in r["cells"]) for t in T}
    add(f"  {'original':<14}" + "".join(f"{number(baseline['cells'][t]):>13}" for t in T))
    add(f"  {'best':<14}" + "".join(f"{number(best_overall[t]):>13}" for t in T))
    add(f"  {'gain':<14}"
        + "".join(f"{number(best_overall[t] - baseline['cells'][t]):>13}" for t in T))

if missing:
    add("")
    add('Missing rows:')
    for label, problem in missing:
        add(f"  {label} — {problem}")

text_path = os.path.join(RESULTS_DIR, "retrieval_variants.txt")
with open(text_path, "w") as handle:
    handle.write("\n".join(lines) + "\n")

tsv = [['Variant', 'Test language'] + [LABELS[t] for t in T]]
for row in table:
    tsv.append([row["label"], row["test_language"] or ""]
               + [number(row["cells"].get(t)) for t in T])
tsv_path = os.path.join(RESULTS_DIR, "retrieval_variants.tsv")
with open(tsv_path, "w") as handle:
    handle.write("\n".join("\t".join(r) for r in tsv) + "\n")

json_path = os.path.join(RESULTS_DIR, "retrieval_variants.json")
with open(json_path, "w") as handle:
    json.dump({"generated": datetime.now().isoformat(timespec="seconds"),
               "paper_baseline": PAPER, "rows": table,
               "missing": [{"label": l, "reason": p} for l, p in missing]},
              handle, indent=2, ensure_ascii=False)

print(f"wrote: {os.path.relpath(text_path, HERE)}")
print(f"wrote: {os.path.relpath(tsv_path, HERE)}")
print(f"wrote: {os.path.relpath(json_path, HERE)}")
