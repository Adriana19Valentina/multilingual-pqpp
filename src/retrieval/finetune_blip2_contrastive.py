import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess

import retrieval_data as rd

CORPUS_CACHE = os.path.join(rd.languages.EMBED_DIR, "corpus_embeddings_blip2.npy")
TEXT_ONLY = ("Qformer.bert.embeddings.word_embeddings.weight",
             "Qformer.bert.embeddings.position_embeddings.weight")

parser = argparse.ArgumentParser(
    description="Contrastive fine-tuning of BLIP-2's text path. The full corpus "
                "does not fit alongside the gradient graph, so negatives are "
                "sampled; validation uses a fixed gallery.")
parser.add_argument("--language", default=rd.languages.TARGET_LANGUAGE,
                    choices=rd.target_languages())
parser.add_argument("--init-from", default="distilled",
                    help="'distilled' (the checkpoint of distil_blip2_text.py), "
                         "'pretrained', or a checkpoint path")
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--negatives", type=int, default=16384)
parser.add_argument("--gallery", type=int, default=30000)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--no-english", dest="also_english", action="store_false")
args = parser.parse_args()
OUT = rd.checkpoint_path("blip2", "contrastive", args.language)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor", model_type="pretrain_vitL",
    is_eval=True, device=device)
init = (rd.checkpoint_path("blip2", "distilled", args.language)
        if args.init_from == "distilled" else args.init_from)
if init != "pretrained":
    if not os.path.exists(init):
        raise SystemExit(f"{init} not found; run distil_blip2_text.py first "
                         f"or pass --init-from pretrained")
    state = torch.load(init, map_location="cpu", weights_only=False)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state["text_state"].items():
            params[name].copy_(value.to(device))
    print(f"initialised from {os.path.basename(init)}")
for name, p in model.named_parameters():
    p.requires_grad = name in TEXT_ONLY
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"trainable: {sum(p.numel() for p in trainable) / 1e6:.1f} M (text path only)")

if not os.path.exists(CORPUS_CACHE):
    raise SystemExit(f"{CORPUS_CACHE} is missing; run reconstruct_retrieval_blip2.py first")
corpus = np.load(CORPUS_CACHE, mmap_mode="r")
image_ids, _ = rd.corpus_image_ids()
assert len(image_ids) == len(corpus), f"{len(image_ids)} ids vs {len(corpus)} embeddings"
row_of = {int(v): i for i, v in enumerate(image_ids)}
print(f"corpus on disk: {corpus.shape}")

relevance = rd.load_relevance()
data = {}
for split in ["train", "val"]:
    data[split] = rd.judged_queries(args.language, split, row_of, relevance)
    print(f"{split}: {len(data[split])} queries, "
          f"{sum(len(e[2]) for e in data[split])} positive pairs")

rng = np.random.default_rng(0)
val_positives = sorted({i for _, _, p in data["val"] for i in p})
taken = set(val_positives)
pool = [i for i in range(len(corpus)) if i not in taken]
gallery_rows = np.array(sorted(
    val_positives + list(rng.choice(pool, size=max(0, args.gallery - len(val_positives)),
                                    replace=False))))
gallery_index = {int(v): i for i, v in enumerate(gallery_rows)}
gallery = F.normalize(
    torch.from_numpy(np.ascontiguousarray(corpus[gallery_rows])).float().to(device),
    dim=-1)
print(f"validation gallery: {tuple(gallery.shape)} "
      f"({len(val_positives)} relevant + random fill)")


def encode(texts, grad=False, batch=64):
    out = []
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        for start in range(0, len(texts), batch):
            processed = [txt_processors["eval"](t) for t in texts[start:start + batch]]
            tokens = model.tokenizer(processed, return_tensors="pt", padding=True,
                                     truncation=True, max_length=128).to(device)
            output = model.Qformer.bert(tokens.input_ids,
                                        attention_mask=tokens.attention_mask,
                                        return_dict=True)
            out.append(F.normalize(output.last_hidden_state[:, 0, :].float(), dim=-1))
    return torch.cat(out)


def validate():
    scores = {}
    for which, index in [("target", 0), ("english", 1)]:
        queries = encode([entry[index] for entry in data["val"]])
        p10 = []
        with torch.no_grad():
            for start in range(0, len(queries), 32):
                chunk = queries[start:start + 32]
                similarity = torch.einsum("bd,ntd->bnt", chunk, gallery).max(dim=2).values
                ranked = similarity.topk(10, dim=1).indices.cpu().numpy()
                for offset, row in enumerate(ranked):
                    wanted = [gallery_index[i] for i in data["val"][start + offset][2]]
                    p10.append(np.isin(row, wanted).mean())
        scores[which] = float(np.mean(p10))
    return scores


baseline = validate()
print(f"\nbefore: val P@10 (gallery)  {args.language}={baseline['target']:.4f}  "
      f"english={baseline['english']:.4f}", flush=True)

optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
best, history = None, []
order = np.arange(len(data["train"]))

for epoch in range(1, args.epochs + 1):
    np.random.shuffle(order)
    total = 0.0
    for start in range(0, len(order), args.batch_size):
        batch = [data["train"][i] for i in order[start:start + args.batch_size]]
        chosen = [p[np.random.randint(len(p))] for _, _, p in batch]
        forbidden = {i for _, _, p in batch for i in p}
        negatives = rng.choice(len(corpus), size=args.negatives + len(forbidden),
                               replace=False)
        negatives = [i for i in negatives if i not in forbidden][:args.negatives]
        rows = np.array(chosen + negatives)
        candidates = F.normalize(
            torch.from_numpy(np.ascontiguousarray(corpus[rows])).float().to(device),
            dim=-1)
        labels = torch.arange(len(batch), device=device)

        texts = [entry[0] for entry in batch]
        if args.also_english:
            texts = texts + [entry[1] for entry in batch]
        queries = encode(texts, grad=True)

        def nce(block):
            logits = torch.einsum("bd,ntd->bnt", block, candidates).max(dim=2).values
            return F.cross_entropy(logits / 0.07, labels)

        loss = nce(queries[:len(batch)])
        if args.also_english:
            loss = (loss + nce(queries[len(batch):])) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss) * len(batch)
        del candidates

    scores = validate()
    history.append({"epoch": epoch, "loss": total / len(order), **scores})
    flag = ""
    if best is None or scores["target"] > best:
        best, flag = scores["target"], "  <-"
        os.makedirs(rd.CHECKPOINT_DIR, exist_ok=True)
        torch.save({"text_state": {name: p.detach().cpu().clone()
                                   for name, p in model.named_parameters()
                                   if name in TEXT_ONLY},
                    "language": args.language, "baseline": baseline,
                    "best_val_p10": best, "history": history, "args": vars(args)}, OUT)
    print(f"epoch {epoch:>2}/{args.epochs}  loss={total / len(order):.4f}  "
          f"val P@10 {args.language}={scores['target']:.4f}  "
          f"english={scores['english']:.4f}{flag}", flush=True)

print(f"\nval P@10 {args.language}: {baseline['target']:.4f} -> {best:.4f}")
print(f"saved {os.path.relpath(OUT, rd.languages.REPO)}")
