import os

# ---------------------------------------------------------------------------
# Uncomment exactly one block below and comment out the rest.
#
#   TARGET_LANGUAGE     the variant every model is trained and reported on, and
#                       the name of the results folder: results/<language>/
#   TRANSLATION_FILE    the file you downloaded, placed in data/. It supplies
#                       the translations AND the train/val/test assignment,
#                       through its `split` column. Nothing is ever written back
#                       to it. Set it to None only when the columns are already
#                       inside data/pqpp_multilingual_*.csv, as for Romanian.
#   RAW_TARGET_LANGUAGE the unreviewed machine translation, if your file has
#                       that column too. It is evaluated but never trained on,
#                       and adds one comparison row to the tables. Leave it None
#                       when you only have the reviewed column.
#   COLUMNS / LABELS    every language you declare here must be non-empty for a
#                       prompt to be used, so declare only the ones you report.
#
# To add a raw machine-translation column to a block, give RAW_TARGET_LANGUAGE
# the language name and add the pair to COLUMNS and LABELS, as the Romanian
# block shows.
# ---------------------------------------------------------------------------

# --- Romanian (already run; results in results/romanian_reviewed/) ---
# TARGET_LANGUAGE = "romanian_reviewed"

# TRANSLATION_FILE = None    # already in data/pqpp_multilingual_*.csv

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

# TRANSLATION_FILE = "final_reviewed_french.csv"

# RAW_TARGET_LANGUAGE = None

# COLUMNS = {
#     "english": "caption",
#     "french_reviewed": "caption_french_reviewed",
# }

# LABELS = {
#     "english": "English",
#     "french_reviewed": "French (reviewed)",
# }

# --- Italian ---
TARGET_LANGUAGE = "italian_reviewed"

TRANSLATION_FILE = "final_reviewed_italian.csv"

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

# TRANSLATION_FILE = "final_reviewed_hindi.csv"

# RAW_TARGET_LANGUAGE = None

# COLUMNS = {
#     "english": "caption",
#     "hindi_reviewed": "caption_hindi_reviewed",
# }

# LABELS = {
#     "english": "English",
#     "hindi_reviewed": "Hindi (reviewed)",
# }

# --- Danish ---
# TARGET_LANGUAGE = "danish_reviewed"

# TRANSLATION_FILE = "final_reviewed_danish.csv"

# RAW_TARGET_LANGUAGE = None

# COLUMNS = {
#     "english": "caption",
#     "danish_reviewed": "caption_danish_reviewed",
# }

# LABELS = {
#     "english": "English",
#     "danish_reviewed": "Danish (reviewed)",
# }

# --- Arabic ---
# TARGET_LANGUAGE = "arabic_reviewed"

# TRANSLATION_FILE = "final_reviewed_arabic.csv"

# RAW_TARGET_LANGUAGE = None

# COLUMNS = {
#     "english": "caption",
#     "arabic_reviewed": "caption_arabic_reviewed",
# }

# LABELS = {
#     "english": "English",
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


def translation_path():
    if TRANSLATION_FILE is None:
        return None
    if os.path.isabs(TRANSLATION_FILE):
        return TRANSLATION_FILE
    return os.path.join(DATA_DIR, TRANSLATION_FILE)


SPLIT_ALIASES = {"train": "train", "training": "train",
                 "val": "val", "valid": "val", "validation": "val", "dev": "val",
                 "test": "test", "testing": "test", "eval": "test"}


def _normalise_split(value):
    name = str(value).strip().lower().replace("-", "_")
    return SPLIT_ALIASES.get(name, SPLIT_ALIASES.get(name.split("_")[0], name))


_CACHE = {}


def load_splits():
    """The three splits, with the translations joined on and the split
    assignment taken from TRANSLATION_FILE."""
    import pandas as pd

    if _CACHE:
        return _CACHE

    if TRANSLATION_FILE is None:
        for name in SPLIT_FILES:
            _CACHE[name] = pd.read_csv(split_path(name))
        return _CACHE

    blocks = []
    for name in SPLIT_FILES:
        block = pd.read_csv(split_path(name))
        block["_split"] = name
        block["_order"] = range(len(block))
        blocks.append(block)
    base = pd.concat(blocks, ignore_index=True)
    base = base.drop(columns=[c for c in base.columns
                              if c.startswith("caption_") and c != "caption_id"])

    path = translation_path()
    if not os.path.exists(path):
        raise SystemExit(
            f"TRANSLATION_FILE '{TRANSLATION_FILE}' not found at {path}.\n"
            f"  download it and put it in data/, or set TRANSLATION_FILE in "
            f"src/languages.py to its full path")
    extra = pd.read_csv(path)

    for needed in ("caption_id", "source"):
        if needed not in extra.columns:
            raise SystemExit(f"{TRANSLATION_FILE} has no '{needed}' column")
    if "split" not in extra.columns:
        raise SystemExit(f"{TRANSLATION_FILE} has no 'split' column")

    wanted = [column for language, column in COLUMNS.items()
              if language != PIVOT]
    absent = [c for c in wanted if c not in extra.columns]
    if absent:
        raise SystemExit(
            f"{TRANSLATION_FILE} has no column(s) {absent}, declared in COLUMNS.\n"
            f"  columns in the file: {[c for c in extra.columns if c.startswith('caption')]}")

    extra = extra[["caption_id", "source", "split"] + wanted].drop_duplicates(
        ["caption_id", "source"])
    extra["split"] = extra["split"].map(_normalise_split)
    unknown = sorted(set(extra["split"]) - set(SPLIT_FILES))
    if unknown:
        raise SystemExit(
            f"{TRANSLATION_FILE}: split column holds {unknown}, which is neither "
            f"train, val nor test")

    merged = base.merge(extra, on=["caption_id", "source"], how="left")
    assert len(merged) == len(base), "the join changed the row count"

    unmatched = int(merged["split"].isna().sum())
    if unmatched == len(merged):
        raise SystemExit(
            f"{TRANSLATION_FILE} matched none of the {len(merged)} prompts; "
            f"check that its caption_id and source come from this benchmark")
    if unmatched:
        # prompts absent from the file keep the benchmark's split and an empty
        # translation; the predictors drop them from every language alike
        merged["split"] = merged["split"].fillna(merged["_split"])

    moved = merged[merged["split"] != merged["_split"]]
    if len(moved):
        raise SystemExit(
            f"{TRANSLATION_FILE} assigns {len(moved)} prompts to a different "
            f"split than the benchmark does.\n"
            f"  The retrieval lists and relevance labels that ship with this "
            f"repository are stored per split, in split order, so a different "
            f"assignment would silently misalign them.\n"
            f"  Either use the published assignment, or regenerate those "
            f"artifacts for the new one.")

    out = {}
    for name in SPLIT_FILES:
        block = merged[merged["split"] == name].sort_values("_order")
        out[name] = block.drop(
            columns=["split", "_split", "_order"]).reset_index(drop=True)
    _CACHE.update(out)
    return out


def load_split(split):
    return load_splits()[split]


def missing_column_message(column, language, frame):
    present = [c for c in frame.columns if c.startswith("caption")]
    where = TRANSLATION_FILE or "data/pqpp_multilingual_*.csv"
    return (
        f"column '{column}', declared for language '{language}' in "
        f"src/languages.py, is not in {where}.\n"
        f"  columns actually present: {present}\n"
        f"  fix COLUMNS in src/languages.py to match that list, or point "
        f"TRANSLATION_FILE at the right file"
    )


def usable_mask(frame):
    import pandas as pd

    mask = pd.Series(True, index=frame.index)
    for language, column in COLUMNS.items():
        if column not in frame.columns:
            raise SystemExit(missing_column_message(column, language, frame))
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
    print(f"target language  : {TARGET_LANGUAGE}")
    print(f"pivot language   : {PIVOT}")
    _tp = translation_path()
    print(f"translation file : "
          f"{os.path.relpath(_tp, REPO) if _tp else 'none (columns already in data/)'}")
    print(f"results go to    : {os.path.relpath(RESULTS_DIR, REPO)}")
    print()
    for split, frame in load_splits().items():
        marks = []
        for name in LANGUAGES:
            column = COLUMNS[name]
            empty = int(frame[column].fillna("").astype(str).str.strip().eq("").sum())
            marks.append(f"{name}=ok" if not empty else f"{name}={empty} empty")
        print(f"{split:<8} {len(frame):>5} rows   " + "   ".join(marks))
