import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats

parser = argparse.ArgumentParser()
parser.add_argument(
    "--decimal",
    default="comma",
    choices=["comma", "dot"],
    help='decimal separator in the TSV files; comma by default so they paste straight into a comma-locale spreadsheet, dot for Google Sheets or an English locale',
)
cli_args = parser.parse_args()

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import languages

PIVOT = languages.PIVOT
TARGET = languages.TARGET_LANGUAGE
RAW_TARGET = languages.RAW_TARGET_LANGUAGE

HERE = languages.REPO
RESULTS_DIR = languages.RESULTS_DIR
PREDICTIONS_DIR = languages.PREDICTIONS_DIR

LANGUAGES = [l for l in (PIVOT, RAW_TARGET, TARGET) if l]
TRAIN_LANGUAGES = [PIVOT, TARGET]
LANGUAGE_LABELS = {l: languages.LABELS.get(l, l) for l in LANGUAGES}

ROW_SPECS = [spec for spec in [
    (PIVOT, PIVOT, LANGUAGE_LABELS[PIVOT]),
    (PIVOT, TARGET, "  pivot -> target"),
    (PIVOT, RAW_TARGET, "  pivot -> target (raw MT)") if RAW_TARGET else None,
    (TARGET, TARGET, LANGUAGE_LABELS[TARGET]),
    (TARGET, PIVOT, "  target -> pivot"),
] if spec]

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

PAPER_BASELINE_CLIP = {
    "glide": (0.649, 0.474),
    "sdxl": (0.380, 0.246),
    "clip_p10": (0.473, 0.299),
    "clip_rr": (0.200, 0.149),
    "blip2_p10": (0.498, 0.358),
    "blip2_rr": (0.166, 0.150),
}

PAPER_BASELINE_CORRCNN = {
    "glide": (0.548, 0.393),
    "sdxl": (0.159, 0.107),
    "clip_p10": (0.270, 0.186),
    "clip_rr": (0.189, 0.162),
    "blip2_p10": (0.159, 0.133),
    "blip2_rr": (0.206, 0.158),
}

PREDICTORS = {
    "bert": {
        "label": "Fine-tuned BERT (mBERT)",
        "dir": RESULTS_DIR,
        "baseline": PAPER_BASELINE,
        "baseline_label": "PQPP paper (BERT-cased)",
        "targets": [key for key, _, _ in MAIN_COLUMNS],
    },
    "clip": {
        "label": "Fine-tuned CLIP (XLM-R ViT-B/32)",
        "dir": os.path.join(RESULTS_DIR, "clip"),
        "baseline": PAPER_BASELINE_CLIP,
        "baseline_label": "PQPP paper (Long-CLIP)",
        "targets": ["glide", "sdxl"],
    },
}

CORRCNN_DIR = os.path.join(RESULTS_DIR, "corrcnn")
CORRCNN_TAG = "longclip-b"

def load_corrcnn(target):
    path = os.path.join(CORRCNN_DIR, f"{target}__{CORRCNN_TAG}.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)

SUBSETS = ["total", "mscoco", "drawbench"]

DEGENERATE = {("drawbench", target) for target in
              ["clip_p10", "clip_rr", "blip2_p10", "blip2_rr"]}

def load(target, language, predictor="bert"):
    spec = PREDICTORS[predictor]
    if target not in spec["targets"]:
        return None
    path = os.path.join(spec["dir"], f"{target}__{language}.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)

ALL_RUNS = {
    predictor: {
        (target, language): load(target, language, predictor)
        for target, _, _ in MAIN_COLUMNS
        for language in TRAIN_LANGUAGES
    }
    for predictor in PREDICTORS
}
RUNS = ALL_RUNS["bert"]

def mark(p):
    return "‡" if p < 0.001 else ("†" if p < 0.01 else " ")

def stats(run, test_language, subset):
    if run is None:
        return None
    evaluation = run["evaluations"].get(test_language)
    if evaluation is None or subset not in evaluation:
        return None
    return evaluation[subset]

def render(metrics, statistic, width=9):
    if metrics is None:
        return "--".rjust(width)
    return f"{metrics[statistic]:.3f}{mark(metrics[f'{statistic}_p'])}".rjust(width)

TEST_N = None
for _predictor in ALL_RUNS.values():
    for _run in _predictor.values():
        _entry = ((_run or {}).get("evaluations", {}).get(PIVOT, {}) or {}).get("total")
        if _entry and _entry.get("n"):
            TEST_N = int(_entry["n"])
            break
    if TEST_N:
        break

lines = []
add = lines.append

add("=" * 130)
add('Multilingual PQPP -- pre-generation / pre-retrieval predictor (fine-tuned BERT)')
add(f"generated: {datetime.now():%Y-%m-%d %H:%M}")
add("=" * 130)
add("")
add("The layout follows Table 3 of PQPP (arXiv 2406.04746v2, p. 8).")
add("The baseline row is the paper's 'Fine-tuned BERT' row: bert-base-cased,")
add("pivot language, the full 6080/2040/2080 split and the same targets.")
add('Our runs use bert-base-multilingual-cased.')
if TEST_N is not None and TEST_N != 2080:
    add("")
    add(f"NOTE: evaluated on {TEST_N} of the 2080 test prompts -- the rest are not")
    add("translated in every declared language, so they are dropped from every")
    add("language alike. Absolute values are therefore not comparable with a run")
    add("over the full split; the ratios between rows are.")
add("")
add('‡ p < 0.001   † p < 0.01   against the random baseline.')
add("")

for subset in SUBSETS:
    add("")
    add("-" * 130)
    add(f"SUBSET: {subset}")
    add("-" * 130)
    add(" " * 22 + "".join(f"{model:>18}" for _, model, _ in MAIN_COLUMNS))
    add(" " * 22 + "".join(f"{measure:>18}" for _, _, measure in MAIN_COLUMNS))
    add(" " * 22 + "".join("  Pearson  Kendall" for _ in MAIN_COLUMNS))
    add("-" * 130)

    if subset == "total":
        row = f"{'PQPP paper (BERT-cased)':<22}"
        for target, _, _ in MAIN_COLUMNS:
            pearson, kendall = PAPER_BASELINE[target]
            row += f"{pearson:.3f}‡".rjust(9) + f"{kendall:.3f}‡".rjust(9)
        add(row)
        add("-" * 130)

    for train_language, test_language, label in ROW_SPECS:
        row = f"{label:<22}"
        for target, _, _ in MAIN_COLUMNS:
            metrics = stats(RUNS[(target, train_language)], test_language, subset)
            if (subset, target) in DEGENERATE and metrics is not None:
                row += "n/a".rjust(9) + "n/a".rjust(9)
                continue
            row += render(metrics, "pearson") + render(metrics, "kendall")
        add(row)

        if subset == "total" and train_language == PIVOT and test_language == PIVOT:
            delta = f"{'  ^ delta vs paper':<22}"
            for target, _, _ in MAIN_COLUMNS:
                metrics = stats(RUNS[(target, train_language)], test_language, subset)
                for index, statistic in enumerate(["pearson", "kendall"]):
                    if metrics is None:
                        delta += "--".rjust(9)
                    else:
                        difference = metrics[statistic] - PAPER_BASELINE[target][index]
                        delta += f"{difference:+.3f} ".rjust(9)
            add(delta)

    if subset == "drawbench":
        add("")
        add('n/a: on retrieval, 95-97% of DrawBench targets are exactly zero -- its prompts')
        add('     do not describe MS COCO images, so the correlations are not interpretable.')

add("")
add("")
add("=" * 130)
add('CLIP PREDICTOR (post-generation) -- generation targets only')
add("=" * 130)
add('Long-CLIP from the paper is monolingual English (a 49,408-token BPE vocabulary,')
add('without diacritics). Replaced with xlm-roberta-base-ViT-B-32 (LAION-5B),')
add('which has a multilingual text tower and the same ViT-B/32 image tower.')
add('CLIP stays frozen; only the 1024-512-256-1 MLP head is trained.')
add('The baseline row is NOT directly comparable: the base model differs too.')
add("")
add(" " * 30 + "".join(f"{model + ' ' + measure:>18}"
                       for _, model, measure in MAIN_COLUMNS[:2]))
add(" " * 30 + "".join("  Pearson  Kendall" for _ in MAIN_COLUMNS[:2]))
add("-" * 66)
row = f"{PREDICTORS['clip']['baseline_label']:<30}"
for target, _, _ in MAIN_COLUMNS[:2]:
    pearson, kendall = PAPER_BASELINE_CLIP[target]
    row += f"{pearson:.3f}‡".rjust(9) + f"{kendall:.3f}‡".rjust(9)
add(row)
add("-" * 66)
for train_language, test_language, label in ROW_SPECS:
    row = f"{label.strip() if label.startswith('  ') else label:<30}"
    for target, _, _ in MAIN_COLUMNS[:2]:
        metrics = stats(ALL_RUNS["clip"][(target, train_language)], test_language, "total")
        row += render(metrics, "pearson") + render(metrics, "kendall")
    add(row)

add("")
add('Comparison between predictors on transfer to the target language:')
add(f"  {'target':<10}{'BERT piv->piv':<14}{'BERT piv->tgt':<14}{'kept':<10}"
    f"{'CLIP piv->piv':<14}{'CLIP piv->tgt':<14}{'kept':<10}")
for target, model, measure in MAIN_COLUMNS[:2]:
    cells = [f"  {target:<10}"]
    for predictor in ["bert", "clip"]:
        run = ALL_RUNS[predictor][(target, PIVOT)]
        in_language = stats(run, PIVOT, "total")
        transferred = stats(run, TARGET, "total")
        if in_language is None or transferred is None:
            cells.append(f"{'--':<38}")
            continue
        retained = transferred["pearson"] / in_language["pearson"]
        cells.append(f"{in_language['pearson']:<14.3f}{transferred['pearson']:<14.3f}"
                     f"{retained:<10.0%}")
    add("".join(cells))
add("")
add('Half of the CLIP input, the image embedding, does not depend on language,')
add('so the predictor retains far more signal under translation than BERT does.')

add("")
add("")
add("=" * 130)
add('GENERATION SUMMARY -- all three predictors from Table 3')
add("=" * 130)
add("")
add(" " * 34 + f"{'GLIDE HBPP':>20}{'SDXL HBPP':>20}")
add(" " * 34 + f"{'Pearson':>10}{'Kendall':>10}{'Pearson':>10}{'Kendall':>10}")
add("-" * 74)
add("PQPP paper (encodere originale)")
for label, baseline in [
    ("  Fine-tuned BERT", PAPER_BASELINE),
    ("  Fine-tuned CLIP", PAPER_BASELINE_CLIP),
    ("  Correlation CNN", PAPER_BASELINE_CORRCNN),
]:
    row = f"{label:<34}"
    for target in ["glide", "sdxl"]:
        pearson, kendall = baseline[target]
        row += f"{pearson:>10.3f}{kendall:>10.3f}"
    add(row)
add("")
add("Rulate de noi")

def generative_row(label, getter):
    row = f"{label:<34}"
    for target in ["glide", "sdxl"]:
        metrics = getter(target)
        if metrics is None:
            row += f"{'--':>10}{'--':>10}"
        else:
            row += f"{metrics['pearson']:>10.3f}{metrics['kendall']:>10.3f}"
    add(row)

generative_row('  BERT (mBERT), pivot',
               lambda t: stats(ALL_RUNS["bert"][(t, PIVOT)], PIVOT, "total"))
generative_row("  BERT (mBERT), piv -> tgt",
               lambda t: stats(ALL_RUNS["bert"][(t, PIVOT)], TARGET, "total"))
generative_row("  BERT (mBERT), target lang.",
               lambda t: stats(ALL_RUNS["bert"][(t, TARGET)], TARGET, "total"))
add("")

def load_clip_control(target):
    path = os.path.join(RESULTS_DIR, "clip_longclip-b", f"{target}__english.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)["evaluations"][PIVOT]["total"]

generative_row("  CLIP control (Long-CLIP), EN", load_clip_control)
generative_row('  CLIP (XLM-R), pivot',
               lambda t: stats(ALL_RUNS["clip"][(t, PIVOT)], PIVOT, "total"))
generative_row("  CLIP (XLM-R), piv -> tgt",
               lambda t: stats(ALL_RUNS["clip"][(t, PIVOT)], TARGET, "total"))
generative_row("  CLIP (XLM-R), target lang.",
               lambda t: stats(ALL_RUNS["clip"][(t, TARGET)], TARGET, "total"))
add("")
generative_row("  Correlation CNN (Long-CLIP)",
               lambda t: (load_corrcnn(t) or {}).get("evaluation", {}).get("total"))
add("")
add('Correlation CNN appears once: it does not use the prompt text, so')
add('its result is the same in any language -- there is no transfer row.')
add("")
add('Sensitivity to translation, by the weight of text in the input (GLIDE):')
add(f"  {'predictor':<22}{'input':<18}{'piv->piv':<10}{'piv->tgt':<10}{'kept':<9}")
for label, source, transfer in [
    ("BERT", lambda: stats(ALL_RUNS["bert"][("glide", PIVOT)], PIVOT, "total"),
     lambda: stats(ALL_RUNS["bert"][("glide", PIVOT)], TARGET, "total")),
    ("CLIP", lambda: stats(ALL_RUNS["clip"][("glide", PIVOT)], PIVOT, "total"),
     lambda: stats(ALL_RUNS["clip"][("glide", PIVOT)], TARGET, "total")),
    ("Correlation CNN", lambda: (load_corrcnn("glide") or {}).get("evaluation", {}).get("total"),
     lambda: (load_corrcnn("glide") or {}).get("evaluation", {}).get("total")),
]:
    a, b = source(), transfer()
    kind = {"BERT": 'text only', "CLIP": "text + imagine",
            "Correlation CNN": 'image only'}[label]
    if a is None or b is None:
        add(f"  {label:<22}{kind:<18}--")
        continue
    add(f"  {label:<22}{kind:<18}{a['pearson']:<10.3f}{b['pearson']:<10.3f}"
        f"{b['pearson'] / a['pearson']:<9.0%}")

PAPER_ROWS = {
    "bert": ("Fine-tuned BERT", PAPER_BASELINE),
    "clip": ("Fine-tuned CLIP", PAPER_BASELINE_CLIP),
    "corrcnn": ("Correlation CNN", PAPER_BASELINE_CORRCNN),
}

def _json(*parts):
    path = os.path.join(RESULTS_DIR, *parts)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)

def cell(predictor, train_language, test_language, target, subset="total"):
    generative = target in ("glide", "sdxl")

    if predictor == "bert":
        run = _json(f"{target}__{train_language}.json")
        node = (run or {}).get("evaluations", {}).get(test_language)

    elif predictor in ("clip", "clip_control"):
        directory = "clip" if predictor == "clip" else "clip_longclip-b"
        if generative:
            run = _json(directory, f"{target}__{train_language}.json")
            node = (run or {}).get("evaluations", {}).get(test_language)
        else:
            directory = ("clip_retrieval" if predictor == "clip"
                         else "clip_retrieval_longclip-b")
            run = _json(directory, f"retrieval__{train_language}.json")
            node = (run or {}).get("evaluations", {}).get(test_language, {}).get(target)

    elif predictor == "corrcnn":

        run = (_json("corrcnn", f"{target}__longclip-b.json") if generative
               else _json("corrcnn_retrieval", f"{target}__longclip-b.json"))
        node = (run or {}).get("evaluation")

    else:
        raise ValueError(predictor)

    return (node or {}).get(subset)

FULL_ROWS = [
    ("bert", PIVOT, PIVOT, '  mBERT, pivot'),
    ("bert", PIVOT, TARGET, "  mBERT, piv -> tgt"),
    ("bert", TARGET, TARGET, "  mBERT, target lang."),
    (None, None, None, ""),
    ("clip_control", PIVOT, PIVOT, "  CLIP control (Long-CLIP)"),
    ("clip", PIVOT, PIVOT, '  CLIP XLM-R, pivot'),
    ("clip", PIVOT, TARGET, "  CLIP XLM-R, piv -> tgt"),
    ("clip", TARGET, TARGET, "  CLIP XLM-R, target lang."),
    (None, None, None, ""),
    ("corrcnn", None, None, "  Correlation CNN (language-blind)"),
]

add("")
add("")
add("=" * 130)
add("FULL TABLE 3 -- all predictors, both tasks, full subset")
add("=" * 130)
add("The Long-CLIP controls reproduce the encoder from the paper. Correlation CNN does not")
add('receive the prompt text, so it has a single row: in its')
add('identical for any language.')
add("")
add(" " * 32 + "".join(f"{model + ' ' + measure:>18}" for _, model, measure in MAIN_COLUMNS))
add(" " * 32 + "".join("  Pearson  Kendall" for _ in MAIN_COLUMNS))
add("-" * 140)
for key, (label, baseline) in PAPER_ROWS.items():
    row = f"  {'paper: ' + label:<30}"
    for target, _, _ in MAIN_COLUMNS:
        pearson, kendall = baseline[target]
        row += f"{pearson:>9.3f}{kendall:>9.3f}"
    add(row)
add("-" * 140)
for predictor, train_language, test_language, label in FULL_ROWS:
    if predictor is None:
        add("")
        continue
    row = f"{label:<32}"
    for target, _, _ in MAIN_COLUMNS:
        metrics = cell(predictor, train_language, test_language, target)
        if metrics is None:
            row += f"{'--':>9}{'--':>9}"
        elif metrics.get("degenerate"):
            row += f"{'n/a':>9}{'n/a':>9}"
        else:
            row += (f"{metrics['pearson']:>8.3f}{mark(metrics['pearson_p'])}"
                    f"{metrics['kendall']:>8.3f}{mark(metrics['kendall_p'])}")
    add(row)
add("")
add('Sensitivity to translation, by the weight of text in the input:')
add(f"  {'predictor':<20}{'input':<18}{'target':<12}{'piv->piv':<10}{'piv->tgt':<10}{'kept':<9}")
for predictor, name, kind in [("bert", "mBERT", 'text only'),
                              ("clip", "CLIP XLM-R", "text + imagine"),
                              ("corrcnn", "Correlation CNN", 'image only')]:
    for target in ["glide", "clip_p10", "blip2_p10"]:
        source = cell(predictor, PIVOT, PIVOT, target)
        transfer = cell(predictor, PIVOT, TARGET, target)
        if source is None or transfer is None or not source.get("pearson"):
            continue
        add(f"  {name:<20}{kind:<18}{target:<12}{source['pearson']:<10.3f}"
            f"{transfer['pearson']:<10.3f}{transfer['pearson'] / source['pearson']:<9.0%}")

add("")
add("")
add("-" * 130)
add("CORP LATEX (subset: total)")
add("-" * 130)
for train_language, test_language, label in ROW_SPECS:
    cells = []
    for target, _, _ in MAIN_COLUMNS:
        metrics = stats(RUNS[(target, train_language)], test_language, "total")
        for statistic in ["pearson", "kendall"]:
            if metrics is None:
                cells.append("--")
            else:
                suffix = {"‡": r"$^\ddagger$", "†": r"$^\dagger$", " ": ""}[
                    mark(metrics[f"{statistic}_p"])
                ]
                cells.append(f"{metrics[statistic]:.3f}{suffix}")
    add(f"{label.strip()} & " + " & ".join(cells) + r" \\")

add("")
add("")
add("-" * 130)
add('CROSS-LINGUAL TRANSFER (subset: total) -- row = training language, column = test language')
add("-" * 130)
for target, model, measure in MAIN_COLUMNS:
    add("")
    add(f"  {model} {measure}")
    add(" " * 24 + "".join(f"{LANGUAGE_LABELS[l]:>20}" for l in LANGUAGES))
    add(" " * 24 + "".join("   Pearson   Kendall" for _ in LANGUAGES))
    for train_language in TRAIN_LANGUAGES:
        row = f"  {LANGUAGE_LABELS[train_language]:<22}"
        for test_language in LANGUAGES:
            metrics = stats(RUNS[(target, train_language)], test_language, "total")
            row += render(metrics, "pearson", 10) + render(metrics, "kendall", 10)
        add(row)

def steiger(truth, predictions_a, predictions_b):
    n = len(truth)
    r_a = scipy.stats.pearsonr(truth, predictions_a)[0]
    r_b = scipy.stats.pearsonr(truth, predictions_b)[0]
    r_ab = scipy.stats.pearsonr(predictions_a, predictions_b)[0]
    mean_square = (r_a ** 2 + r_b ** 2) / 2
    f = (1 - r_ab) / (2 * (1 - mean_square))
    h = (1 - f * mean_square) / (1 - mean_square)
    z = (np.arctanh(r_a) - np.arctanh(r_b)) * np.sqrt(
        (n - 3) / (2 * (1 - r_ab) * h)
    )
    return r_a, r_b, float(z), float(2 * (1 - scipy.stats.norm.cdf(abs(z))))

def predictions_for(target, train_language, test_language):
    path = os.path.join(
        PREDICTIONS_DIR, f"{target}__{train_language}__on_{test_language}.csv"
    )
    return pd.read_csv(path) if os.path.exists(path) else None

transfer_tests = {}

add("")
add("")
add("-" * 130)
add('TRANSFER DEGRADATION: pivot in-language vs. pivot applied to the translation')
add("-" * 130)
add('Steiger test for dependent correlations (same test set, same labels).')
add("")
add(f"  {'target':<14}{'piv->piv':<10}{'piv->tgt':<13}{'kept':<10}{'z':<9}{'p':<11}")
for target, model, measure in MAIN_COLUMNS:
    in_language = predictions_for(target, PIVOT, PIVOT)
    transferred = predictions_for(target, PIVOT, TARGET)
    if in_language is None or transferred is None:
        add(f"  {target:<14}--")
        continue
    r_en, r_ro, z, p = steiger(
        in_language["true_score"].values,
        in_language["predicted_score"].values,
        transferred["predicted_score"].values,
    )
    retained = r_ro / r_en if r_en else float("nan")
    add(
        f"  {target:<14}{r_en:<10.3f}{r_ro:<13.3f}{retained:<10.0%}"
        f"{z:<+9.2f}{p:<11.2e}"
    )
    transfer_tests[target] = {
        "pearson_in_language": round(r_en, 4),
        "pearson_transferred": round(r_ro, 4),
        "retained_fraction": round(retained, 4),
        "steiger_z": round(z, 3),
        "steiger_p": p,
    }
add("")
add('  A small p means the transfer loss is real, not sampling noise.')

add("")
add("")
add("-" * 130)
add('PROVENANCE')
add("-" * 130)
add(f"  {'run':<30}{'grid':<8}{'lr':<10}{'wd':<8}{'epoch':<8}{'val MSE':<12}{'precizie':<10}")
for (target, language), run in sorted(RUNS.items()):
    if run is None:
        add(f"  {target + '__' + language:<30}{'NERULAT':<8}")
        continue
    best = run["best_config"]
    grid = "9/9" if run["grid_complete"] else "incompleta"
    add(
        f"  {run['run']:<30}{grid:<8}{best['learning_rate']:<10g}"
        f"{best['weight_decay']:<8g}{best['epoch']:<8}{best['val_mse']:<12.5f}"
        f"{run.get('precision', 'fp32'):<10}"
    )

missing = [f"{t}__{l}" for (t, l), r in RUNS.items() if r is None]
if missing:
    add("")
    add(f"  Missing {len(missing)} runs:")
    for run_name in sorted(missing):
        target, _, language = run_name.partition("__")
        add(f"    python src/bert/finetunedbert_multilingual.py --language {language} --target {target}")

text_path = os.path.join(RESULTS_DIR, "table3_multilingual.txt")
with open(text_path, "w") as handle:
    handle.write("\n".join(lines) + "\n")

export = {
    "generated": datetime.now().isoformat(timespec="seconds"),
    "source_paper": "PQPP, arXiv 2406.04746v2, Table 3 (p. 8)",
    "split": {"train": 6080, "val": 2040, "test": 2080},
    "test_prompts_evaluated": TEST_N,
    "backbone": "bert-base-multilingual-cased",
    "paper_baseline": {
        "backbone": "bert-base-cased",
        "language": PIVOT,
        "note": 'all values marked ‡ (p < 0.001) in the paper',
        "values": {
            target: {"pearson": pearson, "kendall": kendall}
            for target, (pearson, kendall) in PAPER_BASELINE.items()
        },
    },
    "transfer_significance": {
        "comparison": f"{PIVOT} in-language vs {PIVOT} -> {TARGET} (zero-shot)",
        "test": "Steiger/Williams, dependent correlations",
        "per_target": {},
    },
    "degenerate_subsets": {
        "drawbench": {
            "targets": sorted(target for _, target in DEGENERATE),
            "reason": '95-97% of targets are exactly zero; the correlations are not interpretable',
        }
    },
    "runs": {},
}

for (target, language), run in sorted(RUNS.items()):
    if run is None:
        continue
    entry = {
        "target": target,
        "train_language": language,
        "grid_complete": run["grid_complete"],
        "param_grid": run.get("param_grid"),
        "precision": run.get("precision", "fp32"),
        "best_config": {
            key: run["best_config"][key]
            for key in ["learning_rate", "weight_decay", "epoch", "val_mse", "val_r2"]
            if key in run["best_config"]
        },
        "test": {},
    }
    for test_language in LANGUAGES:
        evaluation = run["evaluations"].get(test_language)
        if evaluation is None:
            continue
        entry["test"][test_language] = {
            "setting": evaluation["setting"],
            **{subset: evaluation[subset] for subset in SUBSETS if subset in evaluation},
        }
    if language == PIVOT:
        entry["delta_vs_paper"] = {
            statistic: round(
                run["evaluations"][PIVOT]["total"][statistic]
                - PAPER_BASELINE[target][index],
                4,
            )
            for index, statistic in enumerate(["pearson", "kendall"])
        }
    export["runs"][run["run"]] = entry

export["transfer_significance"]["per_target"] = transfer_tests

export["full_table"] = {
    "columns": [{"target": t, "model": m, "measure": ms} for t, m, ms in MAIN_COLUMNS],
    "rows": [],
}
for predictor, train_language, test_language, label in FULL_ROWS:
    if predictor is None:
        continue
    export["full_table"]["rows"].append({
        "predictor": predictor,
        "train_language": train_language,
        "test_language": test_language,
        "label": label.strip(),
        "cells": {
            target: cell(predictor, train_language, test_language, target)
            for target, _, _ in MAIN_COLUMNS
        },
    })

export["corrcnn_runs"] = {}
for target in ["glide", "sdxl"]:
    run = load_corrcnn(target)
    if run is None:
        continue
    export["corrcnn_runs"][run["run"]] = {
        "predictor": "correlation_cnn",
        "image_encoder": run["image_encoder"],
        "language_independent": run["language_independent"],
        "note": run["note"],
        "grid_complete": run["grid_complete"],
        "best_config": run["best_config"],
        "test": run["evaluation"],
        "delta_vs_paper": {
            statistic: round(
                run["evaluation"]["total"][statistic]
                - PAPER_BASELINE_CORRCNN[target][index], 4)
            for index, statistic in enumerate(["pearson", "kendall"])
        },
    }

export["clip_runs"] = {}
for (target, language), run in sorted(ALL_RUNS["clip"].items()):
    if run is None:
        continue
    entry = {
        "predictor": "finetuned_clip",
        "backbone": run.get("model_name"),
        "target": target,
        "train_language": language,
        "normalize": run.get("normalize"),
        "images_per_prompt": run.get("images_per_prompt"),
        "grid_complete": run.get("grid_complete"),
        "best_config": run.get("best_config"),
        "test": {
            test_language: {
                "setting": evaluation["setting"],
                **{s_: evaluation[s_] for s_ in SUBSETS if s_ in evaluation},
            }
            for test_language, evaluation in run["evaluations"].items()
        },
    }
    if language == PIVOT:
        entry["delta_vs_paper"] = {
            statistic: round(
                run["evaluations"][PIVOT]["total"][statistic]
                - PAPER_BASELINE_CLIP[target][index], 4)
            for index, statistic in enumerate(["pearson", "kendall"])
        }
    export["clip_runs"][f"{target}__{language}"] = entry

json_path = os.path.join(RESULTS_DIR, "table3_multilingual.json")
with open(json_path, "w") as handle:
    json.dump(export, handle, indent=2, ensure_ascii=False)

def number(value, decimals=3):
    if value is None:
        return ""
    text = f"{value:.{decimals}f}" if decimals is not None else repr(value)
    return text.replace(".", ",") if cli_args.decimal == "comma" else text

def scientific(value):
    if value is None:
        return ""
    text = f"{value:.3e}"
    return text.replace(".", ",") if cli_args.decimal == "comma" else text

wide_header = ["Rand"]
for _, model, measure in MAIN_COLUMNS:
    wide_header += [f"{model} {measure} Pearson", f"{model} {measure} Kendall"]

wide_rows = [wide_header]

baseline_row = ["PQPP paper (BERT-cased)"]
for target, _, _ in MAIN_COLUMNS:
    pearson, kendall = PAPER_BASELINE[target]
    baseline_row += [number(pearson), number(kendall)]
wide_rows.append(baseline_row)

for train_language, test_language, label in ROW_SPECS:
    row = [label.strip()]
    for target, _, _ in MAIN_COLUMNS:
        metrics = stats(RUNS[(target, train_language)], test_language, "total")
        row += [
            number(metrics["pearson"]) if metrics else "",
            number(metrics["kendall"]) if metrics else "",
        ]
    wide_rows.append(row)

    if train_language == PIVOT and test_language == PIVOT:
        delta_row = ["delta vs paper"]
        for target, _, _ in MAIN_COLUMNS:
            metrics = stats(RUNS[(target, train_language)], test_language, "total")
            for index, statistic in enumerate(["pearson", "kendall"]):
                delta_row.append(
                    number(metrics[statistic] - PAPER_BASELINE[target][index])
                    if metrics
                    else ""
                )
        wide_rows.append(delta_row)

wide_path = os.path.join(RESULTS_DIR, "table3_multilingual_wide.tsv")
with open(wide_path, "w") as handle:
    handle.write("\n".join("\t".join(row) for row in wide_rows) + "\n")

long_columns = [
    "predictor",
    "run", "train_language", "test_language", "setting", "target", "model",
    "measure", "task", "subset", "n", "pearson", "pearson_p", "kendall",
    "kendall_p", "spearman", "r2", "rmse", "mae", "bias", "semnificatie",
    "degenerat",
]
long_rows = [long_columns]

for predictor_key, target, model_name, measure, train_language in [
    (p, t, m, ms, l)
    for p in PREDICTORS
    for t, m, ms in MAIN_COLUMNS
    for l in TRAIN_LANGUAGES
]:
    task = "generation" if target in ("glide", "sdxl") else 'retrieval'
    if True:
        run = ALL_RUNS[predictor_key][(target, train_language)]
        if run is None:
            continue
        for test_language in LANGUAGES:
            evaluation = run["evaluations"].get(test_language)
            if evaluation is None:
                continue
            for subset in SUBSETS:
                metrics = evaluation.get(subset)
                if metrics is None:
                    continue
                long_rows.append([
                    predictor_key,
                    run["run"], train_language, test_language,
                    evaluation["setting"], target, model_name, measure, task,
                    subset, str(metrics["n"]),
                    number(metrics["pearson"]), scientific(metrics["pearson_p"]),
                    number(metrics["kendall"]), scientific(metrics["kendall_p"]),
                    number(metrics["spearman"]), number(metrics["r2"]),
                    number(metrics["rmse"], 4), number(metrics["mae"], 4),
                    number(metrics["bias_raw"], 4),
                    mark(metrics["pearson_p"]).strip() or "n.s.",
                    "da" if (subset, target) in DEGENERATE else "",
                ])

long_path = os.path.join(RESULTS_DIR, "table3_multilingual_long.tsv")
with open(long_path, "w") as handle:
    handle.write("\n".join("\t".join(row) for row in long_rows) + "\n")

print(f"wrote: {os.path.relpath(text_path, HERE)}  ({len(lines)} lines)")
print(f"wrote: {os.path.relpath(json_path, HERE)}  ({len(export['runs'])} runs)")
print(f"wrote: {os.path.relpath(wide_path, HERE)}  ({len(wide_rows) - 1} rows)")
print(f"wrote: {os.path.relpath(long_path, HERE)}  ({len(long_rows) - 1} rows)")
print(f"  decimal separator: {'comma' if cli_args.decimal == 'comma' else 'dot'}")
