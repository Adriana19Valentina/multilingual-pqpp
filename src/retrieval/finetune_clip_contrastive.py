import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

import retrieval_data as rd

MODEL_ID = "openai/clip-vit-base-patch32"
CORPUS = os.path.join(rd.languages.EMBED_DIR, "corpus_embeddings_clip-vitb32-hf.npz")

parser = argparse.ArgumentParser(
    description="Contrastive fine-tuning of CLIP's text tower on the relevance "
                "judgements of the training split. The vision tower stays "
                "frozen, so the cached corpus remains valid.")
parser.add_argument("--language", default=rd.languages.TARGET_LANGUAGE,
                    choices=rd.target_languages())
parser.add_argument("--init-from", default="distilled",
                    help="'distilled' (the checkpoint of distil_clip_text_tower.py), "
                         "'pretrained', or a checkpoint path")
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--batch-size", type=int, default=48)
parser.add_argument("--lr", type=float, default=5e-6)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--no-english", dest="also_english", action="store_false")
args = parser.parse_args()
OUT = rd.checkpoint_path("clip-vitb32", "contrastive", args.language)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained(MODEL_ID)
tokenizer = CLIPProcessor.from_pretrained(MODEL_ID).tokenizer
init = (rd.checkpoint_path("clip-vitb32", "distilled", args.language)
        if args.init_from == "distilled" else args.init_from)
if init != "pretrained":
    if not os.path.exists(init):
        raise SystemExit(f"{init} not found; run distil_clip_text_tower.py first "
                         f"or pass --init-from pretrained")
    model.load_state_dict(torch.load(init, map_location="cpu", weights_only=False)
                          ["model_state_dict"])
    print(f"initialised from {os.path.basename(init)}")
model.to(device)
for name, p in model.named_parameters():
    p.requires_grad = name.startswith(("text_model", "text_projection"))
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"trainable: {sum(p.numel() for p in trainable) / 1e6:.1f} M")

if not os.path.exists(CORPUS):
    raise SystemExit(f"{CORPUS} is missing; run reconstruct_retrieval_clip.py first")
cached = np.load(CORPUS)
corpus = F.normalize(torch.from_numpy(cached["embeddings"]).to(device), dim=-1)
row_of = {int(v): i for i, v in enumerate(cached["image_id"])}
print(f"corpus: {tuple(corpus.shape)} (frozen vision tower, from cache)")

relevance = rd.load_relevance()
data = {}
for split in ["train", "val"]:
    data[split] = rd.judged_queries(args.language, split, row_of, relevance)
    print(f"{split}: {len(data[split])} queries, "
          f"{sum(len(e[2]) for e in data[split])} positive pairs")

logit_scale = model.logit_scale.exp().detach()


def encode(texts, grad=False, batch=256):
    out = []
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        for start in range(0, len(texts), batch):
            tokens = tokenizer(texts[start:start + batch], return_tensors="pt",
                               padding=True, truncation=True, max_length=77).to(device)
            out.append(F.normalize(model.get_text_features(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"]).float(), dim=-1))
    return torch.cat(out)


def validate():
    model.eval()
    scores = {}
    for which, index in [("target", 0), ("english", 1)]:
        queries = encode([entry[index] for entry in data["val"]])
        p10 = []
        with torch.no_grad():
            for start in range(0, len(queries), 256):
                ranked = (queries[start:start + 256] @ corpus.T).topk(10, dim=1).indices
                for offset, row in enumerate(ranked.cpu().numpy()):
                    p10.append(np.isin(row, data["val"][start + offset][2]).mean())
        scores[which] = float(np.mean(p10))
    model.train()
    return scores


baseline = validate()
print(f"\nbefore: val P@10  {args.language}={baseline['target']:.4f}  "
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
        labels = torch.tensor(chosen, device=device)
        mask = torch.zeros((len(batch), len(corpus)), dtype=torch.bool, device=device)
        for b, (_, _, positives) in enumerate(batch):
            mask[b, torch.tensor(positives, device=device)] = True
            mask[b, chosen[b]] = False

        texts = [entry[0] for entry in batch]
        if args.also_english:
            texts = texts + [entry[1] for entry in batch]
        queries = encode(texts, grad=True)

        def nce(block):
            logits = (logit_scale * block @ corpus.T).masked_fill(mask, float("-inf"))
            return F.cross_entropy(logits, labels)

        loss = nce(queries[:len(batch)])
        if args.also_english:
            loss = (loss + nce(queries[len(batch):])) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss) * len(batch)

    scores = validate()
    history.append({"epoch": epoch, "loss": total / len(order), **scores})
    flag = ""
    if best is None or scores["target"] > best:
        best, flag = scores["target"], "  <-"
        os.makedirs(rd.CHECKPOINT_DIR, exist_ok=True)
        torch.save({"model_state_dict": {k: v.detach().cpu().clone()
                                         for k, v in model.state_dict().items()},
                    "model_id": MODEL_ID, "language": args.language,
                    "baseline": baseline, "best_val_p10": best,
                    "history": history, "args": vars(args)}, OUT)
    print(f"epoch {epoch:>2}/{args.epochs}  loss={total / len(order):.4f}  "
          f"val P@10 {args.language}={scores['target']:.4f}  "
          f"english={scores['english']:.4f}{flag}", flush=True)

print(f"\nval P@10 {args.language}: {baseline['target']:.4f} -> {best:.4f}")
print(f"saved {os.path.relpath(OUT, rd.languages.REPO)}")
