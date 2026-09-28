import json
import os
from datetime import datetime

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import languages

HERE = languages.DATA_DIR

ENCODERS = {
    "xlmr-vitb32": ("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),
    "xlmr-vith14": ("xlm-roberta-large-ViT-H-14", "frozen_laion5b_s13b_b90k"),
}

import argparse as _argparse
_parser = _argparse.ArgumentParser()
_parser.add_argument("--encoder", default="xlmr-vitb32", choices=sorted(ENCODERS))
_parser.add_argument("--batch-size", type=int, default=64)
_cli = _parser.parse_args()
TAG = _cli.encoder
MODEL_NAME, PRETRAINED = ENCODERS[TAG]

MSCOCO_DIR = languages.IMAGES_DIR
DRAWBENCH_DIR = languages.DRAWBENCH_DIR
OUT_DIR = languages.EMBED_DIR

SPLIT_FILES = {
    "train": "pqpp_multilingual_train.csv",
    "val": "pqpp_multilingual_val.csv",
    "test": "pqpp_multilingual_test.csv",
}

SUFFIXES = {
    "mscoco": {"sdxl": ["_4", "_5"], "glide": ["_7", "_8"]},
    "drawbench": {"sdxl": ["_4", "_5"], "glide": ["_6", "_7"]},
}

DRAWBENCH_ID_OFFSET = 10000

BATCH_SIZE = _cli.batch_size
NUM_WORKERS = 8

os.makedirs(OUT_DIR, exist_ok=True)

def folder_for(caption_id, source):
    if source == "mscoco":
        return os.path.join(MSCOCO_DIR, str(caption_id))
    if source == "drawbench":
        return os.path.join(DRAWBENCH_DIR, str(caption_id + DRAWBENCH_ID_OFFSET))
    raise ValueError(f"sursa necunoscuta: {source}")

splits = {
    name: pd.read_csv(os.path.join(HERE, filename))
    for name, filename in SPLIT_FILES.items()
}

index_rows = []
for split_name, frame in splits.items():
    for _, row in frame.iterrows():
        caption_id, source = int(row["caption_id"]), row["source"]
        folder = folder_for(caption_id, source)

        for generator in ["sdxl", "glide"]:
            for suffix in SUFFIXES[source][generator]:
                index_rows.append(
                    {
                        "split": split_name,

                        "caption_id": caption_id,
                        "source": source,
                        "prompt_key": f"{source}:{caption_id}",
                        "generator": generator,
                        "image_file": f"image{suffix}.png",
                        "path": os.path.join(folder, f"image{suffix}.png"),
                    }
                )

index = pd.DataFrame(index_rows)

index["prompt_index"] = index.groupby(["split", "prompt_key"], sort=False).ngroup()

groups = index.groupby("prompt_index").size()
assert (groups == 4).all(), f"prompts with a different image count: {groups[groups != 4]}"
print(f"images to process: {len(index)}  ({len(groups)} prompts x 4)")
print(index.groupby(["source", "generator"]).size().to_string())

missing = [path for path in index["path"] if not os.path.exists(path)]
if missing:
    raise SystemExit(
        f"missing {len(missing)} images, first: {missing[0]}\n"
        f"Check {MSCOCO_DIR} and {DRAWBENCH_DIR}."
    )
print('all files present\n')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED
)
model.eval().to(device)
print(f"{MODEL_NAME} / {PRETRAINED} pe {device}")

class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        with Image.open(self.paths[idx]) as image:
            return preprocess(image.convert("RGB"))

loader = DataLoader(
    ImageDataset(list(index["path"])),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

chunks = []
with torch.no_grad():
    for step, batch in enumerate(loader, start=1):
        features = model.encode_image(batch.to(device, non_blocking=True))
        chunks.append(features.float().cpu().numpy())
        if step % 50 == 0 or step == len(loader):
            done = min(step * BATCH_SIZE, len(index))
            print(f"  {done}/{len(index)}  ({100 * done / len(index):.0f}%)", flush=True)

embeddings = np.concatenate(chunks).astype(np.float32)
assert embeddings.shape[0] == len(index), "number of embeddings differs from the index"
print(f"\nembeddings: {embeddings.shape}  {embeddings.nbytes / 2**20:.0f} MiB")

norms = np.linalg.norm(embeddings, axis=1)
print(f"L2 norm: min={norms.min():.2f} mean={norms.mean():.2f} max={norms.max():.2f}")

npz_path = os.path.join(OUT_DIR, f"image_embeddings_{TAG}.npz")
np.savez_compressed(
    npz_path,
    embeddings=embeddings,
    split=index["split"].to_numpy(),
    prompt_index=index["prompt_index"].to_numpy(dtype=np.int32),
    prompt_key=index["prompt_key"].to_numpy(),
    caption_id=index["caption_id"].to_numpy(dtype=np.int32),
    source=index["source"].to_numpy(),
    generator=index["generator"].to_numpy(),
    image_file=index["image_file"].to_numpy(),
)

manifest = {
    "created": datetime.now().isoformat(timespec="seconds"),
    "model": {
        "name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "library": f"open_clip_torch {open_clip.__version__}",
        "embed_dim": int(embeddings.shape[1]),
        "context_length_text": int(model.context_length),
        "normalized": False,
    },
    "counts": {
        "images": int(len(index)),
        "prompts": int(index["caption_id"].nunique()),
        "per_source": index.groupby("source").size().to_dict(),
        "per_generator": index.groupby("generator").size().to_dict(),
        "per_split": index.groupby("split").size().to_dict(),
    },
    "layout": {
        "order": 'rows follow the split order (train, val, test); per prompt: sdxl, sdxl, glide, glide',
        "images_per_prompt": 4,
        "grouping_key": 'prompt_index (0..N-1). Do NOT group by caption_id alone: mscoco and drawbench number captions independently and collide (caption_id=81 exists in both, in the test split). The equivalent key is (split, source, caption_id), i.e. prompt_key.',
        "suffixes": SUFFIXES,
        "drawbench_id_offset": DRAWBENCH_ID_OFFSET,
    },
    "usage": {
        "text_embeddings": 'MUST be produced with the same model and pretrained version; a different CLIP lives in a different embedding space',
        "predictor": 'concatenate [text(512) ; image(512)] -> MLP 1024-512-256-1; CLIP stays frozen',
        "aggregation": "at test time, average the 4 predictions of a prompt",
    },
}
manifest_path = os.path.join(OUT_DIR, f"image_embeddings_{TAG}.manifest.json")
with open(manifest_path, "w") as handle:
    json.dump(manifest, handle, indent=2, ensure_ascii=False)

size_mb = os.path.getsize(npz_path) / 2**20
print(f"\nwrote: {os.path.relpath(npz_path, HERE)}  ({size_mb:.0f} MiB)")
print(f"wrote: {os.path.relpath(manifest_path, HERE)}")
