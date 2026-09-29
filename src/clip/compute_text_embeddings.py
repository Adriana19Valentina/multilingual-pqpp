import json
import os
from datetime import datetime

import numpy as np
import open_clip
import pandas as pd
import torch

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import languages

HERE = languages.DATA_DIR

ENCODERS = {
    "xlmr-vitb32": ("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),
    "xlmr-vith14": ("xlm-roberta-large-ViT-H-14", "frozen_laion5b_s13b_b90k"),
}
ENCODERS[f"xlmr-vitb32-ft-{languages.TARGET_LANGUAGE}"] = ("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k")

import argparse as _argparse
_parser = _argparse.ArgumentParser()
_parser.add_argument("--encoder", default="xlmr-vitb32", choices=sorted(ENCODERS))
_cli = _parser.parse_args()
TAG = _cli.encoder
MODEL_NAME, PRETRAINED = ENCODERS[TAG]

OUT_DIR = languages.EMBED_DIR
SPLIT_FILES = languages.SPLIT_FILES
LANGUAGE_COLUMNS = languages.COLUMNS

BATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
if "-ft-" in TAG:

    _lang = TAG.split("-ft-")[1]
    _ck = torch.load(os.path.join(languages.CHECKPOINT_ROOT, "clip",
                                  f"clip_finetuned_{_lang}.pt"),
                     map_location="cpu", weights_only=False)
    model.load_state_dict(_ck["model_state_dict"])
    print(f"adapted weights: epoch {_ck['epoch']}, R@1 "
          f"{_ck['baseline_recall@1']:.3f} -> {_ck['recall@1']:.3f}")
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model.eval().to(device)
print(f"{MODEL_NAME} / {PRETRAINED} on {device}")
print(f"context length: {model.context_length}")

frames = []
for split_name in SPLIT_FILES:
    frame = languages.load_split(split_name)
    frame["split"] = split_name
    frames.append(frame)
data = pd.concat(frames, ignore_index=True)
data["prompt_key"] = data["source"] + ":" + data["caption_id"].astype(str)
print(f"prompts: {len(data)}")

image_path = os.path.join(OUT_DIR, f"image_embeddings_{TAG}.npz")
if os.path.exists(image_path):
    images = np.load(image_path, allow_pickle=True)
    image_keys = pd.DataFrame(
        {
            "prompt_index": images["prompt_index"],
            "prompt_key": images["prompt_key"].astype(str),
            "split": images["split"].astype(str),
        }
    ).drop_duplicates("prompt_index").sort_values("prompt_index")
    assert len(image_keys) == len(data), 'prompt count differs from the images'
    assert (image_keys["prompt_key"].to_numpy() == data["prompt_key"].to_numpy()).all(), (
        'prompt order does not match the image file'
    )
    assert (image_keys["split"].to_numpy() == data["split"].to_numpy()).all()
    print("alignment with image_embeddings verified")
else:
    print('WARNING: image_embeddings is missing; alignment cannot be verified')

def encode(texts):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            tokens = tokenizer(texts[start : start + BATCH_SIZE]).to(device)
            outputs.append(model.encode_text(tokens).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)

arrays = {
    "prompt_index": np.arange(len(data), dtype=np.int32),
    "prompt_key": data["prompt_key"].to_numpy(),
    "split": data["split"].to_numpy(),
    "caption_id": data["caption_id"].to_numpy(dtype=np.int32),
    "source": data["source"].to_numpy(),
}

for language, column in LANGUAGE_COLUMNS.items():
    if column not in data.columns:
        raise SystemExit(languages.missing_column_message(column, language, data))
    text = data[column].fillna("").astype(str)
    blank = int((text.str.strip() == "").sum())
    if blank:
        print(f"  {language}: {blank}/{len(text)} prompts without a translation; "
              f"encoded as empty text and skipped by every predictor")
    embeddings = encode(list(text))
    arrays[f"text_{language}"] = embeddings
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  {language:<20} {embeddings.shape}  mean norm {norms.mean():.2f}")

PIVOT_EMBEDDINGS = arrays[f"text_{languages.PIVOT}"]
EMBED_DIM = int(PIVOT_EMBEDDINGS.shape[1])
pivot_unit = PIVOT_EMBEDDINGS / np.linalg.norm(PIVOT_EMBEDDINGS, axis=1, keepdims=True)
print('\nalignment against the pivot (mean cosine, first 1000 translated prompts):')
for language in LANGUAGE_COLUMNS:
    if language == languages.PIVOT:
        continue
    filled = np.flatnonzero(
        data[LANGUAGE_COLUMNS[language]].fillna("").astype(str).str.strip() != "")[:1000]
    if len(filled) < 2:
        print(f"  {language:<20} no translated prompt to compare")
        continue
    other = arrays[f"text_{language}"]
    other_unit = other / np.linalg.norm(other, axis=1, keepdims=True)
    left, right = pivot_unit[filled], other_unit[filled]
    matched = (left * right).sum(-1).mean()
    cross = left @ right.T
    mismatched = cross[~np.eye(len(filled), dtype=bool)].mean()
    print(f"  {language:<20} matching pairs {matched:.3f}   "
          f"mismatched {mismatched:.3f}   over {len(filled)} prompts")

npz_path = os.path.join(OUT_DIR, f"text_embeddings_{TAG}.npz")
np.savez_compressed(npz_path, **arrays)

manifest = {
    "created": datetime.now().isoformat(timespec="seconds"),
    "model": {
        "name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "library": f"open_clip_torch {open_clip.__version__}",
        "embed_dim": EMBED_DIM,
        "context_length": int(model.context_length),
        "normalized": False,
    },
    "languages": LANGUAGE_COLUMNS,
    "prompts": int(len(data)),
    "alignment_key": 'prompt_index, the same as in image_embeddings; row i of text_<language> corresponds to the 4 images with prompt_index == i',
}
manifest_path = os.path.join(OUT_DIR, f"text_embeddings_{TAG}.manifest.json")
with open(manifest_path, "w") as handle:
    json.dump(manifest, handle, indent=2, ensure_ascii=False)

print(f"\nwrote: {os.path.relpath(npz_path, HERE)} "
      f"({os.path.getsize(npz_path) / 2**20:.0f} MiB)")
print(f"wrote: {os.path.relpath(manifest_path, HERE)}")
