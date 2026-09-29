import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EMBED_DIR = os.path.join(HERE, "clip_embeddings")
GT_DIR = os.path.join(ROOT, "dataset", "retrieval", "ground_truth")

MODEL_ID = "openai/clip-vit-base-patch32"
TOP_K = 25
WIDE_K = 2000
PRECISION_AT = 10
BATCH_SIZE = 256
NUM_WORKERS = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained(MODEL_ID).eval().to(device)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
image_processor = processor.image_processor

with open(os.path.join(EMBED_DIR, "test2017_ids.txt")) as handle:
    test_ids = {int(line) for line in handle if line.strip()}
entries = sorted(
    (int(m.group(1)), os.path.join(HERE, "all", f))
    for f in os.listdir(os.path.join(HERE, "all"))
    if (m := re.match(r"0*(\d+)\.jpg$", f)) and int(m.group(1)) not in test_ids
)
image_ids = np.array([i for i, _ in entries], dtype=np.int64)
paths = [p for _, p in entries]
assert len(paths) == 118287, f"corpus neasteptat: {len(paths)}"
print(f"corpus: {len(paths)} images (train2017)")

class ImageDataset(Dataset):
    def __len__(self):
        return len(paths)

    def __getitem__(self, idx):
        with Image.open(paths[idx]) as image:
            return image_processor(image.convert("RGB"), return_tensors="pt")[
                "pixel_values"
            ][0]

loader = DataLoader(
    ImageDataset(), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)
chunks = []
with torch.no_grad():
    for step, batch in enumerate(loader, start=1):
        features = model.get_image_features(pixel_values=batch.to(device, non_blocking=True))
        chunks.append(features.float().cpu())
        if step % 100 == 0 or step == len(loader):
            print(f"  {min(step * BATCH_SIZE, len(paths))}/{len(paths)}", flush=True)

corpus = torch.nn.functional.normalize(torch.cat(chunks), dim=-1)
np.savez(
    os.path.join(EMBED_DIR, "corpus_embeddings_clip-vitb32-hf.npz"),
    embeddings=corpus.numpy(), image_id=image_ids,
)
corpus = corpus.to(device)
print(f"corpus embeddings: {tuple(corpus.shape)}")

relevance = {}
for split in ["train", "val", "test"]:
    for item in pickle.load(open(os.path.join(GT_DIR, f"retrieval_{split}_gt.pickle"), "rb")):
        relevance[(split, item["source"], int(item["index"]))] = {int(x) for x in item["gt"]}

lists = {}
report = {}
print(f"\n{'split':<6} {'metric':<16} {'identical':>9} {'r':>8}   means")
for split in ["train", "val", "test"]:
    queries_frame = pd.read_csv(
        os.path.join(ROOT, "dataset", "generative", "ground_truth", "average",
                     f"average_{split}.csv")
    )
    reference = pd.read_csv(
        os.path.join(GT_DIR, "clip", f"clip_retrieval_{split}_results.csv")
    )
    captions = list(queries_frame["caption"].astype(str))
    text_chunks = []
    with torch.no_grad():
        for start in range(0, len(captions), 256):
            tokens = processor(
                text=captions[start : start + 256], return_tensors="pt",
                padding=True, truncation=True,
            ).to(device)
            text_chunks.append(
                model.get_text_features(
                    input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
                ).float()
            )
    queries = torch.nn.functional.normalize(torch.cat(text_chunks), dim=-1)

    top_k, precisions, reciprocals = [], [], []
    with torch.no_grad():
        for start in range(0, len(queries), 64):
            ranked = (queries[start : start + 64] @ corpus.T).topk(WIDE_K, dim=1).indices
            ranked = ranked.cpu().numpy()
            for offset, row in enumerate(ranked):
                position = start + offset
                relevant = relevance.get(
                    (split, queries_frame.iloc[position]["source"],
                     int(reference.iloc[position]["index"])), set()
                )
                retrieved = image_ids[row]
                top_k.append(retrieved[:TOP_K])
                precisions.append(
                    np.isin(retrieved[:PRECISION_AT], list(relevant)).mean()
                )
                hits = np.flatnonzero(np.isin(retrieved, list(relevant)))
                reciprocals.append(1.0 / (hits[0] + 1) if len(hits) else 0.0)

    lists[split] = np.array(top_k, dtype=np.int64)
    for name, computed, expected in [
        ("precision", np.array(precisions), reference["precision"].to_numpy()),
        ("reciprocal_rank", np.array(reciprocals), reference["reciprocal_rank"].to_numpy()),
    ]:
        exact = float(np.isclose(computed, expected, atol=1e-9).mean())
        correlation = float(np.corrcoef(computed, expected)[0, 1])
        print(f"{split:<6} {name:<16} {exact:>8.1%} {correlation:>8.4f}   "
              f"{computed.mean():.4f} vs {expected.mean():.4f}")
        report[f"{split}_{name}"] = {
            "exact_match": exact, "pearson": correlation,
            "mean_reconstructed": float(computed.mean()),
            "mean_published": float(expected.mean()),
        }

overall = float(np.mean([v["exact_match"] for v in report.values()]))
print(f"\nexact match, mean: {overall:.1%}")

if overall < 0.95:
    raise SystemExit('Validation failed; the lists are not saved.')

out = os.path.join(EMBED_DIR, "retrieval_lists_clip.npz")
np.savez_compressed(out, **{f"{s}_top{TOP_K}": v for s, v in lists.items()},
                    **{f"{s}_ids": image_ids for s in ["corpus"]})
with open(os.path.join(EMBED_DIR, "retrieval_lists_clip.manifest.json"), "w") as handle:
    json.dump(
        {"model": MODEL_ID, "library": "transformers (preprocesare CLIPImageProcessor)",
         "corpus": "MS COCO train2017, 118287 images", "top_k": TOP_K,
         "validation": report, "overall_exact_match": overall,
         "note": 'the query embeddings in retrieval_process/clip/ are Long-CLIP-L and do NOT correspond to the benchmark'},
        handle, indent=2, ensure_ascii=False,
    )
print(f"wrote: {os.path.relpath(out, HERE)}")
