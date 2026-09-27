import argparse
import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
T = ["clip_p10", "clip_rr", "blip2_p10", "blip2_rr"]
LABELS = {"clip_p10": "CLIP P@10", "clip_rr": "CLIP RR",
          "blip2_p10": "BLIP-2 P@10", "blip2_rr": "BLIP-2 RR"}
PAPER = {"clip_p10": 0.473, "clip_rr": 0.200, "blip2_p10": 0.498, "blip2_rr": 0.166}

ROWS = [
    ("— cod original si controale —", None, None, None),
    ("PQPP paper (Long-CLIP)", None, "PAPER", None),
    ("control Long-CLIP, agregare a", "clip_retrieval_longclip-b", "retrieval__english", "english"),
    ("control Long-CLIP, agregare b", "clip_retrieval_longclip-b", "retrieval__english__b", "english"),
    ("XLM-R, concat, agregare a", "clip_retrieval", "retrieval__english", "english"),
    ("XLM-R, concat, agregare b", "clip_retrieval", "retrieval__english__b", "english"),

    ("— trasaturi de interactiune —", None, None, None),
    ("interactiune, EN → EN", "clip_retrieval", "retrieval__english__b__inter", "english"),
    ("interactiune, EN → RO", "clip_retrieval", "retrieval__english__b__inter", "romanian_reviewed"),
    ("interactiune, RO → RO", "clip_retrieval", "retrieval__romanian_reviewed__b__inter", "romanian_reviewed"),

    ("— turn de text antrenabil —", None, None, None),
    ("inter + turn text, EN → EN", "clip_retrieval", "retrieval__english__b__inter__tt", "english"),
    ("inter + turn text, EN → RO", "clip_retrieval", "retrieval__english__b__inter__tt", "romanian_reviewed"),
    ("inter + turn text, RO → RO (secv.)", "clip_retrieval",
     "retrieval__romanian_reviewed__b__inter__tt__seq", "romanian_reviewed"),

    ("— augmentare multilingva —", None, None, None),
    ("augm EN+RO → EN", "clip_retrieval", "retrieval__english+romanian_reviewed__b__inter", "english"),
    ("augm EN+RO → RO", "clip_retrieval", "retrieval__english+romanian_reviewed__b__inter", "romanian_reviewed"),
    ("augm EN+ambele → EN", "clip_retrieval",
     "retrieval__english+romanian+romanian_reviewed__b__inter", "english"),
    ("augm EN+ambele → RO", "clip_retrieval",
     "retrieval__english+romanian+romanian_reviewed__b__inter", "romanian_reviewed"),
    ("augm EN+RO + consistenta → RO", "clip_retrieval",
     "retrieval__english+romanian_reviewed__b__inter__cons0.5", "romanian_reviewed"),

    ("— secvential (continuare din EN) —", None, None, None),
    ("secv. RO → RO", "clip_retrieval", "retrieval__romanian_reviewed__b__inter__seq", "romanian_reviewed"),
    ("secv. + augm EN+RO → RO", "clip_retrieval",
     "retrieval__english+romanian_reviewed__b__inter__seq", "romanian_reviewed"),
    ("secv. + augm ambele → RO", "clip_retrieval",
     "retrieval__english+romanian+romanian_reviewed__b__inter__seq", "romanian_reviewed"),
    ("secv. + augm + turn text → EN", "clip_retrieval",
     "retrieval__english+romanian_reviewed__b__inter__tt__seq", "english"),
    ("secv. + augm + turn text → RO", "clip_retrieval",
     "retrieval__english+romanian_reviewed__b__inter__tt__seq", "romanian_reviewed"),
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
add("Predictor post-regasire: variantele testate")
add(f"generat: {datetime.now():%Y-%m-%d %H:%M}")
add("=" * 92)
add("")
add("Encoder XLM-R ViT-B/32 (inghetat, daca nu se spune altfel). Listele de")
add("regasire sunt reconstruite si validate contra tintelor publicate.")
add("EN → RO inseamna: antrenat pe engleza, testat pe romana revizuita.")
add("")
add(f"{'varianta':<38}" + "".join(f"{LABELS[t]:>13}" for t in T))
add("-" * 92)

table, missing = [], []
for label, directory, name, language in ROWS:
    if name is None:
        add("")
        add(label)
        continue
    values, problem = cells(directory, name, language)
    if values is None:
        add(f"  {label:<36}" + f"{'(' + (problem or 'lipsa') + ')':>13}")
        missing.append((label, problem or "lipsa"))
        continue
    add(f"  {label:<36}" + "".join(f"{number(values.get(t)):>13}" for t in T))
    table.append({"label": label, "run": name, "test_language": language,
                  "cells": values})

add("")
add("-" * 92)
add("Cea mai buna varianta pe fiecare tinta:")
for t in T:
    best = max((row for row in table if t in row["cells"]),
               key=lambda row: row["cells"][t])
    add(f"  {LABELS[t]:<14}{number(best['cells'][t]):>8}   {best['label']}")

add("")
add("Castigul fata de codul original (concat, agregare a, doar engleza):")
baseline = next((r for r in table if r["run"] == "retrieval__english"
                 and r["test_language"] == "english"), None)
if baseline:
    best_overall = {t: max(r["cells"][t] for r in table if t in r["cells"]) for t in T}
    add(f"  {'original':<14}" + "".join(f"{number(baseline['cells'][t]):>13}" for t in T))
    add(f"  {'cel mai bun':<14}" + "".join(f"{number(best_overall[t]):>13}" for t in T))
    add(f"  {'castig':<14}"
        + "".join(f"{number(best_overall[t] - baseline['cells'][t]):>13}" for t in T))

if missing:
    add("")
    add("Randuri lipsa:")
    for label, problem in missing:
        add(f"  {label} — {problem}")

text_path = os.path.join(RESULTS_DIR, "retrieval_variants.txt")
with open(text_path, "w") as handle:
    handle.write("\n".join(lines) + "\n")

tsv = [["Varianta", "Limba test"] + [LABELS[t] for t in T]]
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

print(f"scris: {os.path.relpath(text_path, HERE)}")
print(f"scris: {os.path.relpath(tsv_path, HERE)}")
print(f"scris: {os.path.relpath(json_path, HERE)}")
