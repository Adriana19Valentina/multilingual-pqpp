import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import retrieval_data as rd

TOP_K = 25
WIDE_K = 2000
PRECISION_AT = 10
CHECKPOINT_NAMES = {"clip": "clip-vitb32", "xlmr": "xlmr-vitb32", "blip2": "blip2"}
OUTPUT_NAMES = {"clip": "clip", "xlmr": "xlmr-vitb32", "blip2": "blip2"}

parser = argparse.ArgumentParser(
    description="Retrieval over MS COCO train2017 with queries in one language, "
                "scored against the published relevance judgements.")
parser.add_argument("--system", required=True, choices=sorted(CHECKPOINT_NAMES))
parser.add_argument("--model", default="pretrained",
                    choices=["pretrained", "distilled", "contrastive"])
parser.add_argument("--language", default=rd.languages.TARGET_LANGUAGE,
                    choices=rd.languages.LANGUAGES)
parser.add_argument("--splits", nargs="+", default=rd.SPLITS, choices=rd.SPLITS)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--num-workers", type=int, default=12)
args = parser.parse_args()

TAG = OUTPUT_NAMES[args.system] + ("" if args.model == "pretrained" else f"-{args.model}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = None
if args.model != "pretrained":
    path = rd.checkpoint_path(CHECKPOINT_NAMES[args.system], args.model, args.language)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; train it first")
    state = torch.load(path, map_location="cpu", weights_only=False)
    detail = (f"cosine {state['baseline']['cosine']:.4f} -> {state['best_cosine']:.4f}"
              if "best_cosine" in state else
              f"val P@10 {state['baseline']['target']:.4f} -> {state['best_val_p10']:.4f}")
    print(f"text tower from {os.path.basename(path)}: {detail}")

image_ids, paths = rd.corpus_image_ids()

if args.system == "clip":
    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    if state:
        model.load_state_dict(state["model_state_dict"])
    model.eval().to(device)
    tokenizer = CLIPProcessor.from_pretrained(model_name).tokenizer
    cache = os.path.join(rd.languages.EMBED_DIR, "corpus_embeddings_clip-vitb32-hf.npz")
    if not os.path.exists(cache):
        raise SystemExit(f"{cache} is missing; run reconstruct_retrieval_clip.py first")
    cached = np.load(cache)
    assert (cached["image_id"] == image_ids).all(), "cached corpus is out of order"
    corpus = F.normalize(torch.from_numpy(cached["embeddings"]).to(device), dim=-1)

    def encode_text(texts):
        tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=77).to(device)
        return model.get_text_features(input_ids=tokens["input_ids"],
                                       attention_mask=tokens["attention_mask"])

elif args.system == "xlmr":
    import open_clip

    model_name, pretrained = "xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    if state:
        model.load_state_dict(state["model_state_dict"])
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    cache = os.path.join(rd.languages.EMBED_DIR, "corpus_embeddings_xlmr-vitb32.npz")

    class ImageDataset(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, idx):
            with Image.open(paths[idx]) as image:
                return preprocess(image.convert("RGB"))

    if os.path.exists(cache):
        cached = np.load(cache)
        assert (cached["image_id"] == image_ids).all(), "cached corpus is out of order"
        corpus = torch.from_numpy(cached["embeddings"]).to(device)
    else:
        print(f"encoding {len(paths)} images (cached afterwards)", flush=True)
        loader = DataLoader(ImageDataset(), batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)
        chunks = []
        with torch.no_grad():
            for step, batch in enumerate(loader, start=1):
                chunks.append(model.encode_image(
                    batch.to(device, non_blocking=True)).float().cpu())
                if step % 50 == 0 or step == len(loader):
                    print(f"  {min(step * args.batch_size, len(paths))}/{len(paths)}",
                          flush=True)
        corpus = F.normalize(torch.cat(chunks), dim=-1)
        np.savez(cache, embeddings=corpus.numpy(), image_id=image_ids)
        corpus = corpus.to(device)
    corpus = F.normalize(corpus, dim=-1)

    def encode_text(texts):
        return model.encode_text(tokenizer(texts).to(device))

else:
    from lavis.models import load_model_and_preprocess

    model_name = "blip2_feature_extractor / pretrain_vitL"
    model, _, txt_processors = load_model_and_preprocess(
        name="blip2_feature_extractor", model_type="pretrain_vitL",
        is_eval=True, device=device)
    if state:
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in state["text_state"].items():
                params[name].copy_(value.to(device))
    cache = os.path.join(rd.languages.EMBED_DIR, "corpus_embeddings_blip2.npy")
    if not os.path.exists(cache):
        raise SystemExit(f"{cache} is missing; run reconstruct_retrieval_blip2.py first")
    corpus = np.load(cache, mmap_mode="r")
    assert len(corpus) == len(image_ids), f"{len(image_ids)} ids vs {len(corpus)} embeddings"

    def encode_text(texts):
        processed = [txt_processors["eval"](t) for t in texts]
        return model.extract_features({"text_input": processed}, mode="text") \
            .text_embeds[:, 0, :]

print(f"{model_name} on {device}, queries: {args.language}", flush=True)

frames, queries = {}, {}
for split in args.splits:
    frame, texts = rd.query_texts(args.language, split)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), 128):
            chunks.append(encode_text(texts[start:start + 128]).float().cpu())
    frames[split], queries[split] = frame, F.normalize(torch.cat(chunks), dim=-1)

if args.system == "blip2":
    del model
    torch.cuda.empty_cache()
    gpu_corpus = torch.empty((len(corpus), *corpus.shape[1:]), dtype=torch.float32,
                             device=device)
    for start in range(0, len(corpus), 8192):
        block = torch.from_numpy(np.ascontiguousarray(corpus[start:start + 8192]))
        gpu_corpus[start:start + len(block)] = F.normalize(block.to(device).float(), dim=-1)
    corpus = gpu_corpus


def rank(block):
    if args.system == "blip2":
        similarity = torch.einsum("bd,ntd->bnt", block, corpus).max(dim=2).values
    else:
        similarity = block @ corpus.T
    return similarity.topk(WIDE_K, dim=1).indices.cpu().numpy()


relevance = rd.load_relevance()
reference_system = "blip2" if args.system == "blip2" else "clip"
report, per_query = {}, {}
print(f"\n{'split':<6} {'metric':<16} {'ours':>9} {'English':>9} {'kept':>8}", flush=True)
for split in args.splits:
    frame = frames[split]
    reference = rd.reference_results(reference_system, split)
    index = rd.reference_results("clip", split)["index"]
    top_k, precisions, reciprocals = [], [], []
    step = 8 if args.system == "blip2" else 64
    with torch.no_grad():
        for start in range(0, len(frame), step):
            ranked = rank(queries[split][start:start + step].to(device))
            for offset, row in enumerate(ranked):
                position = start + offset
                relevant = list(relevance.get(
                    (split, frame["source"].iloc[position], int(index.iloc[position])), ()))
                retrieved = image_ids[row]
                top_k.append(retrieved[:TOP_K])
                precisions.append(np.isin(retrieved[:PRECISION_AT], relevant).mean())
                hits = np.flatnonzero(np.isin(retrieved, relevant))
                reciprocals.append(1.0 / (hits[0] + 1) if len(hits) else 0.0)

    per_query[split] = {"precision": np.array(precisions),
                        "reciprocal_rank": np.array(reciprocals),
                        f"top{TOP_K}": np.array(top_k, dtype=np.int64)}
    report[split] = {}
    for name in ["precision", "reciprocal_rank"]:
        ours = per_query[split][name]
        english = reference[name].to_numpy()
        kept = ours.mean() / english.mean() if english.mean() else float("nan")
        print(f"{split:<6} {name:<16} {ours.mean():>9.4f} {english.mean():>9.4f} "
              f"{kept:>7.1%}", flush=True)
        report[split][name] = {"mean": float(ours.mean()),
                               "mean_english_reference": float(english.mean()),
                               "kept": float(kept)}

out = os.path.join(rd.languages.EMBED_DIR, f"retrieval_{TAG}_{args.language}.npz")
np.savez_compressed(out, **{f"{s}_{n}": v for s, d in per_query.items()
                            for n, v in d.items()})
os.makedirs(rd.RESULTS_DIR, exist_ok=True)
summary = os.path.join(rd.RESULTS_DIR, f"retrieval_{TAG}_{args.language}.json")
with open(summary, "w") as handle:
    json.dump({"model": model_name, "variant": args.model, "language": args.language,
               "corpus": f"MS COCO train2017, {len(image_ids)} images",
               "reference": f"published {reference_system} English results",
               "precision_at": PRECISION_AT, "rr_depth": WIDE_K,
               "per_split": report}, handle, indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.relpath(out, rd.languages.REPO)}")
print(f"wrote {os.path.relpath(summary, rd.languages.REPO)}")
