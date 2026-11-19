import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import languages

PQPP_DATASET = os.path.join(os.path.dirname(languages.REPO), "dataset")
GT_DIR = os.path.join(PQPP_DATASET, "retrieval", "ground_truth")
CHECKPOINT_DIR = os.path.join(languages.CHECKPOINT_ROOT, "retrieval")
RESULTS_DIR = os.path.join(languages.RESULTS_DIR, "retrieval")
SPLITS = ["train", "val", "test"]
CORPUS_SIZE = 118287


def target_languages():
    return [l for l in languages.LANGUAGES if l != languages.PIVOT]


def checkpoint_path(system, method, language):
    return os.path.join(CHECKPOINT_DIR, f"{system}-{method}-{language}.pt")


def load_relevance():
    relevance = {}
    for split in SPLITS:
        with open(os.path.join(GT_DIR, f"retrieval_{split}_gt.pickle"), "rb") as handle:
            for item in pickle.load(handle):
                relevance[(split, item["source"], int(item["index"]))] = {
                    int(x) for x in item["gt"]}
    return relevance


def reference_results(system, split):
    return pd.read_csv(os.path.join(GT_DIR, system, f"{system}_retrieval_{split}_results.csv"))


def query_texts(language, split):
    frame = languages.load_split(split)
    column = languages.column_for(language)
    if column not in frame.columns:
        raise SystemExit(languages.missing_column_message(column, language, frame))
    texts = frame[column].fillna("").astype(str).str.strip()
    empty = int((texts == "").sum())
    if empty:
        raise SystemExit(f"{language} / {split}: {empty} empty queries; "
                         f"retrieval needs a translation for every prompt")
    return frame, texts.tolist()


def parallel_captions(language, split):
    frame = languages.load_split(split)
    column = languages.column_for(language)
    if column not in frame.columns:
        raise SystemExit(languages.missing_column_message(column, language, frame))
    frame = frame[frame["source"] == "mscoco"]
    target = frame[column].fillna("").astype(str).str.strip()
    frame, target = frame[target != ""], target[target != ""]
    return [str(x) for x in frame["caption"]], target.tolist()


def judged_queries(language, split, row_of, relevance):
    frame = languages.load_split(split)
    column = languages.column_for(language)
    if column not in frame.columns:
        raise SystemExit(languages.missing_column_message(column, language, frame))
    reference = reference_results("clip", split)
    entries = []
    for position in np.flatnonzero((frame["source"] == "mscoco").to_numpy()):
        target = frame[column].iloc[position]
        if pd.isna(target) or not str(target).strip():
            continue
        positives = [row_of[i] for i in relevance.get(
            (split, frame["source"].iloc[position],
             int(reference["index"].iloc[position])), ()) if i in row_of]
        if positives:
            entries.append((str(target), str(frame["caption"].iloc[position]), positives))
    return entries


def corpus_image_ids():
    with open(os.path.join(languages.EMBED_DIR, "test2017_ids.txt")) as handle:
        test_ids = {int(line) for line in handle if line.strip()}
    entries = sorted(
        (int(m.group(1)), os.path.join(languages.CORPUS_DIR, f))
        for f in os.listdir(languages.CORPUS_DIR)
        if (m := re.match(r"0*(\d+)\.jpg$", f)) and int(m.group(1)) not in test_ids)
    assert len(entries) == CORPUS_SIZE, f"expected {CORPUS_SIZE} images, found {len(entries)}"
    return np.array([i for i, _ in entries], dtype=np.int64), [p for _, p in entries]
