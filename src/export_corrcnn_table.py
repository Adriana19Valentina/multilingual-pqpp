import argparse
import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

COLUMNS = [
    ("glide", "GLIDE", "HBPP", "generare"),
    ("sdxl", "SDXL", "HBPP", "generare"),
    ("clip_p10", "CLIP", "P@10", "regasire"),
    ("clip_rr", "CLIP", "RR", "regasire"),
    ("blip2_p10", "BLIP-2", "P@10", "regasire"),
    ("blip2_rr", "BLIP-2", "RR", "regasire"),
]

PAPER = {
    "glide": (0.548, 0.393), "sdxl": (0.159, 0.107),
    "clip_p10": (0.270, 0.186), "clip_rr": (0.189, 0.162),
    "blip2_p10": (0.159, 0.133), "blip2_rr": (0.206, 0.158),
}

SETUP = {
    "generare": {
        "images": 4, "note": "2 SDXL + 2 GLIDE",
        "conv": "3 straturi (16-64)", "fc1": "262144 -> 512", "out": "sigmoid",
    },
    "regasire": {
        "images": 25, "note": "top-25 ale sistemului prezis",
        "conv": "4 straturi (32-64)", "fc1": "65536 -> 1024", "out": "ReLU",
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--decimal", default="comma", choices=["comma", "dot"])
cli = parser.parse_args()

def load(*parts):
    path = os.path.join(RESULTS_DIR, *parts)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)

def run_for(target):
    generative = target in ("glide", "sdxl")
    return load("corrcnn" if generative else "corrcnn_retrieval",
                f"{target}__longclip-b.json")

def cell(target, subset="total"):
    run = run_for(target)
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
add("Correlation CNN -- predictor post-generare / post-regasire")
add(f"generat: {datetime.now():%Y-%m-%d %H:%M}")
add("=" * 118)
add("")
add("Intrarea e o matrice de corelatie 512x512 intre dimensiunile embedding-ului,")
add("calculata peste imaginile promptului. Textul NU intra niciodata in model:")
add("`retrieve_embeddings` exista in codul original, dar nu e apelata.")
add("")
add("De aceea tabelul are un singur rand de rezultate. O varianta 'multilingva'")
add("ar da cifre identice, bit cu bit. Encoderul de imagine e Long-CLIP, cel din")
add("paper, singurul care permite comparatia cu Tabelul 3.")
add("")
add("‡ p < 0.001   † p < 0.01   -- fata de linia de baza aleatoare")
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

row = f"{'Rulat de noi':<32}"
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
add("Pe subseturi (Pearson):")
add(f"  {'tinta':<12}{'total':<10}{'mscoco':<10}{'drawbench':<12}")
for target, _, _, _ in COLUMNS:
    parts = [number(cell(target, s)["pearson"]) if cell(target, s) else "--"
             for s in ("total", "mscoco", "drawbench")]
    add(f"  {target:<12}{parts[0]:<10}{parts[1]:<10}{parts[2]:<12}")

add("")
add("Configuratia difera intre task-uri (asa e si in codul original):")
add(f"  {'':<12}{'imagini':<10}{'convolutii':<22}{'fc1':<18}{'iesire':<10}")
for task, setup in SETUP.items():
    add(f"  {task:<12}{setup['images']:<10}{setup['conv']:<22}"
        f"{setup['fc1']:<18}{setup['out']:<10}")
add("")
add("Numarul de imagini conteaza: matricea are rang cel mult (imagini - 1), deci")
add("pe generare e puternic degenerata (rang <= 3), iar pe regasire mult mai bine")
add("conditionata (rang <= 24).")

add("")
add("Limitare de raportat: codul original impacheteaza cele doua matrici de")
add("regasire ca [2, 512, 512] si foloseste tinta MEDIATA, dar arhitectura")
add("accepta un singur canal, iar Tabelul 3 raporteaza per sistem. Cele doua nu")
add("se pot impaca; s-a ales varianta compatibila cu tabelul -- un model per")
add("(sistem, metrica). De aici probabil si abaterile mai mari pe regasire.")

text_path = os.path.join(RESULTS_DIR, "correlation_cnn_table.txt")
with open(text_path, "w") as handle:
    handle.write("\n".join(lines) + "\n")

header = ["Rand"] + [f"{m} {ms} {s}" for _, m, ms, _ in COLUMNS
                     for s in ("Pearson", "Kendall")]
tsv = [
    header,
    ["PQPP paper"] + [number(v) for target, _, _, _ in COLUMNS for v in PAPER[target]],
    ["Rulat de noi"] + [
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
        "note": "nu foloseste textul promptului; rezultatele sunt identice in orice limba",
        "image_encoder": "longclip-b",
        "paper_baseline": {t: {"pearson": p, "kendall": k} for t, (p, k) in PAPER.items()},
        "setup": SETUP,
        "cells": {t: {s: cell(t, s) for s in ("total", "mscoco", "drawbench")}
                  for t, _, _, _ in COLUMNS},
    }, handle, indent=2, ensure_ascii=False)

print(f"scris: {os.path.relpath(text_path, HERE)}")
print(f"scris: {os.path.relpath(tsv_path, HERE)}")
print(f"scris: {os.path.relpath(json_path, HERE)}")
