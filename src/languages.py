import os

# Uncomment exactly one block and comment out the rest. TARGET_LANGUAGE also
# names the results folder, so each language writes to results/<language>/.
# The column names must match the ones in data/pqpp_multilingual_*.csv.

# --- Romanian (already run; results in results/romanian_reviewed/) ---
# TARGET_LANGUAGE = "romanian_reviewed"

# RAW_TARGET_LANGUAGE = "romanian"

# COLUMNS = {
#     "english": "caption",
#     "romanian": "caption_romanian",
#     "romanian_reviewed": "caption_romanian_reviewed",
# }

# LABELS = {
#     "english": "English",
#     "romanian": "Romanian (raw MT)",
#     "romanian_reviewed": "Romanian (reviewed)",
# }

# --- French ---
# TARGET_LANGUAGE = "french_reviewed"
#
# RAW_TARGET_LANGUAGE = "french"
#
# COLUMNS = {
#     "english": "caption",
#     "french": "caption_french",
#     "french_reviewed": "caption_french_reviewed",
# }
#
# LABELS = {
#     "english": "English",
#     "french": "French (raw MT)",
#     "french_reviewed": "French (reviewed)",
# }

# --- Italian ---
TARGET_LANGUAGE = "italian_reviewed"

RAW_TARGET_LANGUAGE = None

COLUMNS = {
    "english": "caption",
    "italian_reviewed": "caption_italian_reviewed",
}

LABELS = {
    "english": "English",
    "italian_reviewed": "Italian (reviewed)",
}

# --- Hindi ---
# TARGET_LANGUAGE = "hindi_reviewed"
#
# RAW_TARGET_LANGUAGE = "hindi"
#
# COLUMNS = {
#     "english": "caption",
#     "hindi": "caption_hindi",
#     "hindi_reviewed": "caption_hindi_reviewed",
# }
#
# LABELS = {
#     "english": "English",
#     "hindi": "Hindi (raw MT)",
#     "hindi_reviewed": "Hindi (reviewed)",
# }

# --- Danish ---
# TARGET_LANGUAGE = "danish_reviewed"
#
# RAW_TARGET_LANGUAGE = "danish"
#
# COLUMNS = {
#     "english": "caption",
#     "danish": "caption_danish",
#     "danish_reviewed": "caption_danish_reviewed",
# }
#
# LABELS = {
#     "english": "English",
#     "danish": "Danish (raw MT)",
#     "danish_reviewed": "Danish (reviewed)",
# }

# --- Arabic ---
# TARGET_LANGUAGE = "arabic_reviewed"
#
# RAW_TARGET_LANGUAGE = "arabic"
#
# COLUMNS = {
#     "english": "caption",
#     "arabic": "caption_arabic",
#     "arabic_reviewed": "caption_arabic_reviewed",
# }
#
# LABELS = {
#     "english": "English",
#     "arabic": "Arabic (raw MT)",
#     "arabic_reviewed": "Arabic (reviewed)",
# }

PIVOT = "english"

RESULTS_ROOT = "results"
RESULTS_SUBDIR = TARGET_LANGUAGE

LANGUAGES = list(COLUMNS)
TRAIN_LANGUAGES = [PIVOT, TARGET_LANGUAGE]

assert PIVOT in COLUMNS, "PIVOT is not listed in COLUMNS"
assert TARGET_LANGUAGE in COLUMNS, "TARGET_LANGUAGE is not listed in COLUMNS"
assert RAW_TARGET_LANGUAGE is None or RAW_TARGET_LANGUAGE in COLUMNS, \
    "RAW_TARGET_LANGUAGE is neither None nor listed in COLUMNS"
assert all(l in COLUMNS for l in LABELS), "LABELS names a language absent from COLUMNS"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DATA_DIR = os.path.join(REPO, "data")
EMBED_DIR = os.path.join(DATA_DIR, "clip_embeddings")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
DRAWBENCH_DIR = os.path.join(DATA_DIR, "drawbench_images")
CORPUS_DIR = os.path.join(DATA_DIR, "all")
THIRD_PARTY = os.path.join(REPO, "third_party")
CHECKPOINT_ROOT = os.path.join(REPO, "checkpoints")

RESULTS_DIR = os.path.join(REPO, RESULTS_ROOT, RESULTS_SUBDIR)
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")

SPLIT_FILES = {
    "train": "pqpp_multilingual_train.csv",
    "val": "pqpp_multilingual_val.csv",
    "test": "pqpp_multilingual_test.csv",
}


def column_for(language):
    if language not in COLUMNS:
        raise KeyError(
            f"language '{language}' is not defined in src/languages.py; "
            f"available: {LANGUAGES}"
        )
    return COLUMNS[language]


def split_path(split):
    return os.path.join(DATA_DIR, SPLIT_FILES[split])


def usable_mask(frame):
    import pandas as pd

    mask = pd.Series(True, index=frame.index)
    for language, column in COLUMNS.items():
        if column not in frame.columns:
            raise KeyError(
                f"column '{column}' for language '{language}' is missing from the "
                f"CSV; check COLUMNS in src/languages.py"
            )
        text = frame[column]
        mask &= text.notna() & (text.astype(str).str.strip() != "")
    return mask.to_numpy()


def report_coverage(name, mask):
    kept = int(mask.sum())
    if kept == len(mask):
        return kept
    print(f"  {name}: {kept}/{len(mask)} prompts translated in every language, "
          f"{len(mask) - kept} dropped")
    if kept == 0:
        raise SystemExit(
            f"{name}: no prompt is translated in every language declared in "
            f"src/languages.py; remove the languages you do not have"
        )
    return kept


if __name__ == "__main__":
    import pandas as pd

    print(f"target language : {TARGET_LANGUAGE}")
    print(f"pivot language  : {PIVOT}")
    print(f"results go to   : {os.path.relpath(RESULTS_DIR, REPO)}")
    print()
    for split in SPLIT_FILES:
        path = split_path(split)
        if not os.path.exists(path):
            print(f"{split:<8} MISSING {os.path.relpath(path, REPO)}")
            continue
        frame = pd.read_csv(path)
        marks = []
        for name in LANGUAGES:
            column = COLUMNS[name]
            if column not in frame.columns:
                marks.append(f"{name}=ABSENT")
            elif frame[column].isna().any():
                marks.append(f"{name}={int(frame[column].isna().sum())} empty")
            else:
                marks.append(f"{name}=ok")
        print(f"{split:<8} {len(frame):>5} rows   " + "   ".join(marks))
