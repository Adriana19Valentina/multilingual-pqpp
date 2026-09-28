import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EMBED_DIR = os.path.join(HERE, "clip_embeddings")
GT_DIR = os.path.join(ROOT, "dataset", "retrieval", "ground_truth")

TOP_K = 25
WIDE_K = 2000
PRECISION_AT = 10
QUERY_BATCH = 8
BATCH_SIZE = 64
NUM_WORKERS = 12
CORPUS_CACHE = os.path.join(EMBED_DIR, "corpus_embeddings_blip2.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor", model_type="pretrain_vitL",
    is_eval=True, device=device,
)
print("BLIP-2 pretrain_vitL incarcat", flush=True)

with open(os.path.join(EMBED_DIR, "test2017_ids.txt")) as handle:
    test_ids = {int(line) for line in handle if line.strip()}
entries = sorted(
    (int(m.group(1)), os.path.join(HERE, "all", f))
    for f in os.listdir(os.path.join(HERE, "all"))
    if (m := re.match(r"0*(\d+)\.jpg$", f)) and int(m.group(1)) not in test_ids
)
image_ids = np.array([i for i, _ in entries], dtype=np.int64)
paths = [p for _, p in entries]
assert len(paths) == 118287
print(f"corpus: {len(paths)} images", flush=True)

class ImageDataset(Dataset):
    def __len__(self):
        return len(paths)

    def __getitem__(self, idx):
        with Image.open(paths[idx]) as image:
            return vis_processors["eval"](image.convert("RGB"))

if os.path.exists(CORPUS_CACHE):
    corpus = np.load(CORPUS_CACHE, mmap_mode="r")
    print(f"corpus from cache: {corpus.shape}", flush=True)
else:

    corpus = np.lib.format.open_memmap(
        CORPUS_CACHE, mode="w+", dtype=np.float16, shape=(len(paths), 32, 768)
    )
    loader = DataLoader(
        ImageDataset(), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )
    written = 0
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            features = model.extract_features(
                {"image": batch.to(device, non_blocking=True)}, mode="image"
            )

            embeds = features.image_embeds.half().cpu().numpy()
            corpus[written : written + len(embeds)] = embeds
            written += len(embeds)
            if step % 100 == 0 or step == len(loader):
                print(f"  {written}/{len(paths)}", flush=True)
    corpus.flush()
    print(f"corpus embeddings: {corpus.shape}", flush=True)

relevance = {}
for split in ["train", "val", "test"]:
    for item in pickle.load(open(os.path.join(GT_DIR, f"retrieval_{split}_gt.pickle"), "rb")):
        relevance[(split, item["source"], int(item["index"]))] = {int(x) for x in item["gt"]}

query_frames, query_embeddings = {}, {}
for split in ["train", "val", "test"]:
    frame = pd.read_csv(
        os.path.join(ROOT, "dataset", "generative", "ground_truth", "average",
                     f"average_{split}.csv")
    )
    captions = list(frame["caption"].astype(str))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(captions), 128):
            processed = [txt_processors["eval"](c) for c in captions[start : start + 128]]
            features = model.extract_features({"text_input": processed}, mode="text")

            chunks.append(features.text_embeds[:, 0, :].float().cpu())
    query_frames[split] = frame
    query_embeddings[split] = F.normalize(torch.cat(chunks), dim=-1)
    print(f"query {split}: {tuple(query_embeddings[split].shape)}", flush=True)

del model
torch.cuda.empty_cache()

gpu_corpus = torch.empty((len(corpus), 32, 768), dtype=torch.float32, device=device)
for start in range(0, len(corpus), 8192):
    block = torch.from_numpy(np.ascontiguousarray(corpus[start : start + 8192])).to(device)
    gpu_corpus[start : start + len(block)] = F.normalize(block.float(), dim=-1)
    del block
torch.cuda.empty_cache()
print(f"pe GPU: {tuple(gpu_corpus.shape)} {gpu_corpus.dtype}", flush=True)

lists, report = {}, {}
print(f"\n{'split':<6} {'metric':<16} {'identic':>9} {'r':>8}   medii")
for split in ["train", "val", "test"]:
    queries_frame = query_frames[split]
    reference = pd.read_csv(
        os.path.join(GT_DIR, "blip2", f"blip2_retrieval_{split}_results.csv")
    )
    queries = query_embeddings[split].to(device)

    top_k, precisions, reciprocals = [], [], []
    with torch.no_grad():
        for start in range(0, len(queries), QUERY_BATCH):
            chunk = queries[start : start + QUERY_BATCH]

            similarity = torch.einsum("bd,ntd->bnt", chunk, gpu_corpus).max(dim=2).values
            ranked = similarity.topk(WIDE_K, dim=1).indices.cpu().numpy()
            for offset, row in enumerate(ranked):
                position = start + offset
                relevant = relevance.get(
                    (split, queries_frame.iloc[position]["source"],
                     int(reference.iloc[position]["index"])), set()
                )
                retrieved = image_ids[row]
                top_k.append(retrieved[:TOP_K])
                precisions.append(np.isin(retrieved[:PRECISION_AT], list(relevant)).mean())
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
              f"{computed.mean():.4f} vs {expected.mean():.4f}", flush=True)
        report[f"{split}_{name}"] = {
            "exact_match": exact, "pearson": correlation,
            "mean_reconstructed": float(computed.mean()),
            "mean_published": float(expected.mean()),
        }

overall = float(np.mean([v["exact_match"] for v in report.values()]))
print(f"\nexact match, mean: {overall:.1%}")

if overall < 0.95:
    raise SystemExit('Validation failed; the lists are not saved.')

out = os.path.join(EMBED_DIR, "retrieval_lists_blip2.npz")
np.savez_compressed(out, **{f"{s}_top{TOP_K}": v for s, v in lists.items()})
with open(os.path.join(EMBED_DIR, "retrieval_lists_blip2.manifest.json"), "w") as handle:
    json.dump(
        {"model": "LAVIS blip2_feature_extractor / pretrain_vitL",
         "ranking": "cosine over the 32 image tokens, max-pooled, sorted descending",
         "corpus": "MS COCO train2017, 118287 images", "top_k": TOP_K,
         "validation": report, "overall_exact_match": overall},
        handle, indent=2, ensure_ascii=False,
    )
print(f"wrote: {os.path.relpath(out, HERE)}")
