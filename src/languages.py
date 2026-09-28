import os

TARGET_LANGUAGE = "romanian_reviewed"

RAW_TARGET_LANGUAGE = "romanian"

COLUMNS = {
    "english": "caption",
    "romanian": "caption_romanian",
    "romanian_reviewed": "caption_romanian_reviewed",
}

LABELS = {
    "english": "English",
    "romanian": "Romanian (raw MT)",
    "romanian_reviewed": "Romanian (reviewed)",
}

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
