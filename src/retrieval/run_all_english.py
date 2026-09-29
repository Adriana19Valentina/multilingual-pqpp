import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "dataset")
EMBED_DIR = os.path.join(DATA_DIR, "clip_embeddings")
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GT_DIR = os.path.join(ROOT, "dataset", "retrieval", "ground_truth")
OUT_DIR = os.path.join(DATA_DIR, "retrieval_results")

SPLITS = ["train", "val", "test"]

SEARCH_DEPTH = 2000

parser = argparse.ArgumentParser()
parser.add_argument("--system", required=True, choices=["clip", "blip2"])
parser.add_argument("--top-k", type=int, default=100,
                    help='ids and scores saved per query')
parser.add_argument("--precision-at", type=int, default=10)
args = parser.parse_args()

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.system == "clip":

    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    corpus_npz = np.load(os.path.join(EMBED_DIR, "corpus_embeddings_clip-vitb32-hf.npz"))
    image_ids = corpus_npz["image_id"]
    corpus = F.normalize(torch.from_numpy(corpus_npz["embeddings"]).to(device), dim=-1)

    def encode(prompts):
        chunks = []
        with torch.no_grad():
            for start in range(0, len(prompts), 256):
                tokens = processor(text=prompts[start : start + 256], return_tensors="pt",
                                   padding=True, truncation=True).to(device)
                chunks.append(model.get_text_features(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"]).float())
        return F.normalize(torch.cat(chunks), dim=-1)

    def rank(queries, depth):
        ids, scores = [], []
        with torch.no_grad():
            for start in range(0, len(queries), 64):
                top = (queries[start : start + 64] @ corpus.T).topk(depth, dim=1)
                ids.append(image_ids[top.indices.cpu().numpy()])
                scores.append(top.values.cpu().numpy())
        return np.concatenate(ids), np.concatenate(scores)

else:
    from lavis.models import load_model_and_preprocess

    model, _, txt_processors = load_model_and_preprocess(
        name="blip2_feature_extractor", model_type="pretrain_vitL",
        is_eval=True, device=device,
    )

    cache = os.path.join(EMBED_DIR, "corpus_embeddings_blip2.npy")
    if not os.path.exists(cache):
        raise SystemExit(f"missing {cache}; run reconstruct_retrieval_blip2.py first")
    raw = np.load(cache, mmap_mode="r")
    corpus_ids_file = os.path.join(EMBED_DIR, "test2017_ids.txt")
    with open(corpus_ids_file) as handle:
        test_ids = {int(line) for line in handle if line.strip()}
    import re
    entries = sorted(
        int(m.group(1)) for f in os.listdir(os.path.join(DATA_DIR, "all"))
        if (m := re.match(r"0*(\d+)\.jpg$", f)) and int(m.group(1)) not in test_ids
    )
    image_ids = np.array(entries, dtype=np.int64)
    assert len(image_ids) == raw.shape[0]

    corpus = torch.empty((raw.shape[0], 32, 768), dtype=torch.float32, device=device)
    for start in range(0, len(raw), 8192):
        block = torch.from_numpy(np.ascontiguousarray(raw[start : start + 8192])).to(device)
        corpus[start : start + len(block)] = F.normalize(block.float(), dim=-1)
        del block
    torch.cuda.empty_cache()

    def encode(prompts):
        chunks = []
        with torch.no_grad():
            for start in range(0, len(prompts), 128):
                processed = [txt_processors["eval"](p) for p in prompts[start : start + 128]]
                features = model.extract_features({"text_input": processed}, mode="text")
                chunks.append(features.text_embeds[:, 0, :].float())
        return F.normalize(torch.cat(chunks), dim=-1)

    def rank(queries, depth):
        ids, scores = [], []
        with torch.no_grad():
            for start in range(0, len(queries), 8):

                similarity = torch.einsum(
                    "bd,ntd->bnt", queries[start : start + 8], corpus
                ).max(dim=2).values
                top = similarity.topk(depth, dim=1)
                ids.append(image_ids[top.indices.cpu().numpy()])
                scores.append(top.values.cpu().numpy())
        return np.concatenate(ids), np.concatenate(scores)

print(f"sistem={args.system}  corpus={len(image_ids)} images", flush=True)

arrays, score_rows, report = {}, [], {}
for split in SPLITS:
    frame = pd.read_csv(os.path.join(DATA_DIR, f"pqpp_multilingual_{split}.csv"))
    reference = pd.read_csv(
        os.path.join(GT_DIR, args.system, f"{args.system}_retrieval_{split}_results.csv")
    )
    relevance = {
        (item["source"], int(item["index"])): {int(x) for x in item["gt"]}
        for item in pickle.load(
            open(os.path.join(GT_DIR, f"retrieval_{split}_gt.pickle"), "rb")
        )
    }

    queries = encode([str(x) for x in frame["caption"]])
    ids, scores = rank(queries, SEARCH_DEPTH)
    arrays[f"{split}_ids"] = ids[:, : args.top_k]
    arrays[f"{split}_scores"] = scores[:, : args.top_k].astype(np.float32)

    precisions, reciprocals = [], []
    for position in range(len(frame)):
        relevant = list(relevance.get(
            (frame.iloc[position]["source"], int(reference.iloc[position]["index"])), set()
        ))
        retrieved = ids[position]
        precisions.append(np.isin(retrieved[: args.precision_at], relevant).mean())
        hits = np.flatnonzero(np.isin(retrieved, relevant))
        reciprocals.append(1.0 / (hits[0] + 1) if len(hits) else 0.0)

    precisions, reciprocals = np.array(precisions), np.array(reciprocals)
    score_rows.append(pd.DataFrame({
        "split": split, "source": frame["source"], "caption_id": frame["caption_id"],
        "prompt": frame["caption"],
        "precision_computed": precisions, "precision_published": reference["precision"],
        "rr_computed": reciprocals, "rr_published": reference["reciprocal_rank"],
    }))

    for name, computed, expected in [
        ("precision", precisions, reference["precision"].to_numpy()),
        ("reciprocal_rank", reciprocals, reference["reciprocal_rank"].to_numpy()),
    ]:
        exact = float(np.isclose(computed, expected, atol=1e-9).mean())
        report[f"{split}_{name}"] = {
            "exact_match": exact,
            "pearson": float(np.corrcoef(computed, expected)[0, 1]),
            "mean_computed": float(computed.mean()),
            "mean_published": float(expected.mean()),
        }
        print(f"  {split:<6} {name:<16} identic={exact:6.1%}  "
              f"means {computed.mean():.4f} vs {expected.mean():.4f}", flush=True)

npz_path = os.path.join(OUT_DIR, f"{args.system}_english.npz")
np.savez_compressed(npz_path, **arrays, corpus_image_ids=image_ids)
csv_path = os.path.join(OUT_DIR, f"{args.system}_english_scores.csv")
pd.concat(score_rows, ignore_index=True).to_csv(csv_path, index=False)

overall = float(np.mean([v["exact_match"] for v in report.values()]))
with open(os.path.join(OUT_DIR, f"{args.system}_english.manifest.json"), "w") as handle:
    json.dump({
        "system": args.system,
        "model": ("openai/clip-vit-base-patch32 through transformers"
                  if args.system == "clip"
                  else "LAVIS blip2_feature_extractor / pretrain_vitL, spatiu 768"),
        "corpus": "MS COCO train2017, 118287 images",
        "prompts": 'all 10200, in the pivot language',
        "top_k_saved": args.top_k, "search_depth_for_rr": SEARCH_DEPTH,
        "validation": report, "overall_exact_match": overall,
    }, handle, indent=2, ensure_ascii=False)

print(f"\nexact match, mean: {overall:.1%}")
print(f"wrote: {os.path.relpath(npz_path, HERE)}  "
      f"({os.path.getsize(npz_path) / 2**20:.0f} MiB)")
print(f"wrote: {os.path.relpath(csv_path, HERE)}")
