import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATASET = os.path.join(ROOT, "dataset")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

TRANSLATED = {
    "train": "average_train_translated_part1_full.csv",
    "val": "average_val_translated.csv",
    "test": "average_test_translated.csv",
}
EXPECTED_SIZES = {"train": 6080, "val": 2040, "test": 2080}

TRANSLATION_COLUMNS = ["caption_romanian", "caption_romanian_reviewed", "caption_french"]

def key(series):
    return series.astype(str).str.strip()

def generative(model, split):
    path = os.path.join(DATASET, "generative", "ground_truth", model, f"{model}_{split}.csv")
    df = pd.read_csv(path)
    return pd.DataFrame({"_key": key(df["caption"]), f"hbpp_{model}": df["score"].values})

def retrieval(model, split):
    if model == "average":
        path = os.path.join(DATASET, "retrieval", "ground_truth", "average", f"average_{split}.csv")
    else:
        path = os.path.join(
            DATASET, "retrieval", "ground_truth", model, f"{model}_retrieval_{split}_results.csv"
        )
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "_key": key(df["prompt"]),
            f"p10_{model}": df["precision"].values,
            f"rr_{model}": df["reciprocal_rank"].values,
        }
    )

for split, filename in TRANSLATED.items():
    base = pd.read_csv(os.path.join(OUT_DIR, filename))
    assert len(base) == EXPECTED_SIZES[split], f"{split}: {len(base)} rows"

    merged = base[["caption_id", "caption", "source"] + TRANSLATION_COLUMNS].copy()
    merged["_key"] = key(base["caption"])
    assert merged["_key"].is_unique, f"{split}: duplicate prompts, joining on text is not safe"

    sources = (
        [generative(m, split) for m in ["glide", "sdxl", "average"]]
        + [retrieval(m, split) for m in ["clip", "blip2", "average"]]
    )

    for table in sources:
        assert table["_key"].is_unique, f"{split}: chei duplicate intr-o sursa"
        before = len(merged)
        merged = merged.merge(table, on="_key", how="inner", validate="one_to_one")
        assert len(merged) == before, f"{split}: {before - len(merged)} prompts left unmatched"

    merged = merged.rename(
        columns={
            "hbpp_average": "hbpp_average",
            "p10_average": "p10_average",
            "rr_average": "rr_average",
        }
    ).drop(columns="_key")

    assert ((merged.hbpp_glide + merged.hbpp_sdxl) / 2 - merged.hbpp_average).abs().max() < 1e-9
    assert ((merged.p10_clip + merged.p10_blip2) / 2 - merged.p10_average).abs().max() < 1e-9
    assert ((merged.rr_clip + merged.rr_blip2) / 2 - merged.rr_average).abs().max() < 1e-9

    for column in ["hbpp_glide", "hbpp_sdxl", "hbpp_average"]:
        assert merged[column].between(-1, 2).all(), f"{split}: {column} outside [-1, 2]"
    for column in ["p10_clip", "rr_clip", "p10_blip2", "rr_blip2", "p10_average", "rr_average"]:
        assert merged[column].between(0, 1).all(), f"{split}: {column} outside [0, 1]"

    for column in TRANSLATION_COLUMNS:
        assert merged[column].notna().all(), f"{split}: missing translations in {column}"

    assert (merged.caption_id.values == base.caption_id.values).all(), f"{split}: ordine schimbata"
    assert ((merged.hbpp_average - base.score).abs().max() < 1e-9), f"{split}: previous target differs"

    merged["score"] = merged["hbpp_average"]

    out = os.path.join(OUT_DIR, f"pqpp_multilingual_{split}.csv")
    merged.to_csv(out, index=False)
    print(f"{split:<6} n={len(merged):<5} -> {os.path.basename(out)}")
    print(f"       {merged.source.value_counts().to_dict()}")
