import argparse
import json
import os

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import languages

PIVOT = languages.PIVOT
TARGET = languages.TARGET_LANGUAGE
RAW_TARGET = languages.RAW_TARGET_LANGUAGE

HERE = languages.REPO
RESULTS_DIR = languages.RESULTS_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

LANGUAGES = [l for l in (PIVOT, RAW_TARGET, TARGET) if l]
LANGUAGE_LABELS = {l: languages.LABELS.get(l, l) for l in LANGUAGES}

TRAIN_LANGUAGES = [PIVOT, TARGET]

MAIN_COLUMNS = [
    ("glide", "GLIDE", "HBPP"),
    ("sdxl", "SDXL", "HBPP"),
    ("clip_p10", "CLIP", "P@10"),
    ("clip_rr", "CLIP", "RR"),
    ("blip2_p10", "BLIP-2", "P@10"),
    ("blip2_rr", "BLIP-2", "RR"),
]

PAPER_BASELINE = {
    "glide": (0.566, 0.406),
    "sdxl": (0.281, 0.232),
    "clip_p10": (0.451, 0.277),
    "clip_rr": (0.221, 0.176),
    "blip2_p10": (0.511, 0.328),
    "blip2_rr": (0.168, 0.139),
}

ALL_TARGETS = [key for key, _, _ in MAIN_COLUMNS]

parser = argparse.ArgumentParser()
parser.add_argument("--table", default="main", choices=["main", "transfer"])
parser.add_argument("--target", default="glide", choices=ALL_TARGETS,
                    help='only for --table transfer')
parser.add_argument("--subset", default="total", choices=["total", "mscoco", "drawbench"])
parser.add_argument("--latex", action="store_true")
parser.add_argument(
    "--no-baseline",
    action="store_true",
    help='hide the paper baseline row and the deltas against it',
)
args = parser.parse_args()

def load(target, language):
    path = os.path.join(RESULTS_DIR, f"{target}__{language}.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)

def mark(p):
    return "‡" if p < 0.001 else ("†" if p < 0.01 else " ")

def cell(result, test_language, statistic):
    if result is None:
        return None
    evaluation = result["evaluations"].get(test_language)
    if evaluation is None or args.subset not in evaluation:
        return None
    metrics = evaluation[args.subset]
    return metrics[statistic], metrics[f"{statistic}_p"]

def render(value, width=8):
    if value is None:
        return "--".rjust(width)
    correlation, p = value
    return f"{correlation:.3f}{mark(p)}".rjust(width)

def latex(value):
    if value is None:
        return "--"
    correlation, p = value
    suffix = {"‡": r"$^\ddagger$", "†": r"$^\dagger$", " ": ""}[mark(p)]
    return f"{correlation:.3f}{suffix}"

missing = []

if args.table == "main":
    header_model = " " * 22 + "".join(
        f"{model:>18}" for _, model, _ in MAIN_COLUMNS
    )
    header_measure = " " * 22 + "".join(
        f"{measure:>18}" for _, _, measure in MAIN_COLUMNS
    )
    header_stat = " " * 22 + "".join("  Pearson  Kendall" for _ in MAIN_COLUMNS)

    print(f"\nTable 3 (multilingual) -- subset: {args.subset}, in-language evaluation\n")
    print(header_model)
    print(header_measure)
    print(header_stat)
    print("-" * (22 + 18 * len(MAIN_COLUMNS)))

    show_baseline = not args.no_baseline and args.subset == "total"
    if show_baseline:
        line = f"{'PQPP paper (BERT-cased)':<22}"
        for target, _, _ in MAIN_COLUMNS:
            pearson, kendall = PAPER_BASELINE[target]
            line += f"{pearson:.3f}‡".rjust(9) + f"{kendall:.3f}‡".rjust(9)
        print(line)
        print("-" * (22 + 18 * len(MAIN_COLUMNS)))

    rows = []
    for language in TRAIN_LANGUAGES:
        line = f"{LANGUAGE_LABELS[language]:<22}"
        latex_cells = []
        for target, _, _ in MAIN_COLUMNS:
            result = load(target, language)
            if result is None:
                missing.append(f"{target}__{language}")
            elif not result.get("grid_complete", True):
                missing.append(f"{target}__{language} (incomplete grid)")
            pearson = cell(result, language, "pearson")
            kendall = cell(result, language, "kendall")
            line += render(pearson, 9) + render(kendall, 9)
            latex_cells += [latex(pearson), latex(kendall)]
        print(line)
        rows.append((LANGUAGE_LABELS[language], latex_cells))

        if show_baseline and language == PIVOT:
            delta = f"{'  ^ delta vs paper':<22}"
            for target, _, _ in MAIN_COLUMNS:
                result = load(target, language)
                for index, statistic in enumerate(["pearson", "kendall"]):
                    value = cell(result, language, statistic)
                    if value is None:
                        delta += "--".rjust(9)
                    else:
                        delta += f"{value[0] - PAPER_BASELINE[target][index]:+.3f} ".rjust(9)
            print(delta)

    if args.latex:
        print("\n% --- corp de tabel LaTeX ---")
        for label, cells in rows:
            print(f"{label} & " + " & ".join(cells) + r" \\")

else:
    print(
        f"\nCross-lingual transfer -- target: {args.target}, subset: {args.subset}\n"
        f"rows = training language, columns = test language\n"
    )
    header = " " * 22 + "".join(f"{LANGUAGE_LABELS[l]:>20}" for l in LANGUAGES)
    print(header)
    print(" " * 22 + "".join("   Pearson   Kendall" for _ in LANGUAGES))
    print("-" * (22 + 20 * len(LANGUAGES)))

    for train_language in TRAIN_LANGUAGES:
        result = load(args.target, train_language)
        if result is None:
            missing.append(f"{args.target}__{train_language}")
        elif not result.get("grid_complete", True):
            missing.append(f"{args.target}__{train_language} (incomplete grid)")
        line = f"{LANGUAGE_LABELS[train_language]:<22}"
        for test_language in LANGUAGES:
            line += render(cell(result, test_language, "pearson"), 10)
            line += render(cell(result, test_language, "kendall"), 10)
        print(line)

print('\n‡ p < 0.001   † p < 0.01   (against the random baseline)')

if missing:
    print(f"\nTo run ({len(missing)}):")
    for run in sorted(set(missing)):
        target, _, language = run.partition("__")
        note = " (incomplete grid -- rerun)" if "incompleta" in language else ""
        language = language.split(" ")[0]
        print(
            f"  python src/bert/finetunedbert_multilingual.py "
            f"--language {language} --target {target}{note}"
        )
else:
    print("\nEvery cell has been run.")
