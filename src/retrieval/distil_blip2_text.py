import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess

import retrieval_data as rd

TEXT_ONLY = ("Qformer.bert.embeddings.word_embeddings.weight",
             "Qformer.bert.embeddings.position_embeddings.weight")

parser = argparse.ArgumentParser(
    description="Distil the target language into BLIP-2's text path. Only the "
                "token and position embeddings are trained, because the rest of "
                "the Q-Former is shared with the image path.")
parser.add_argument("--language", default=rd.languages.TARGET_LANGUAGE,
                    choices=rd.target_languages())
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--batch-size", type=int, default=96)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--keep-english", type=float, default=0.5)
args = parser.parse_args()
OUT = rd.checkpoint_path("blip2", "distilled", args.language)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor", model_type="pretrain_vitL",
    is_eval=True, device=device)

frozen = {name: p.detach().clone()
          for name, p in model.named_parameters() if name in TEXT_ONLY}
for name, p in model.named_parameters():
    p.requires_grad = name in TEXT_ONLY
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"trainable: {sum(p.numel() for p in trainable) / 1e6:.1f} M "
      f"of {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M "
      f"(text path only; the image path is untouched)")

data = {}
for split in ["train", "val"]:
    data[split] = rd.parallel_captions(args.language, split)
    print(f"{split}: {len(data[split][0])} parallel captions")


def embed(texts, grad=False, batch=128):
    out = []
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        for start in range(0, len(texts), batch):
            processed = [txt_processors["eval"](t) for t in texts[start:start + batch]]
            tokens = model.tokenizer(processed, return_tensors="pt",
                                     padding=True, truncation=True,
                                     max_length=128).to(device)
            output = model.Qformer.bert(tokens.input_ids,
                                        attention_mask=tokens.attention_mask,
                                        return_dict=True)
            out.append(output.last_hidden_state[:, 0, :].float())
    return torch.cat(out)


def with_frozen_text(fn):
    saved = {name: p.detach().clone()
             for name, p in model.named_parameters() if name in TEXT_ONLY}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in TEXT_ONLY:
                p.copy_(frozen[name])
    result = fn()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in TEXT_ONLY:
                p.copy_(saved[name])
    return result


english_train, target_train = data["train"]
english_val, target_val = data["val"]
targets = {
    "train": with_frozen_text(lambda: F.normalize(embed(english_train), dim=-1).cpu()),
    "val": with_frozen_text(lambda: F.normalize(embed(english_val), dim=-1).cpu()),
}


def evaluate():
    gold = targets["val"].to(device)
    theirs = F.normalize(embed(target_val), dim=-1)
    ours_en = F.normalize(embed(english_val), dim=-1)
    aligned = float((theirs * gold).sum(-1).mean())
    drift = float((ours_en * gold).sum(-1).mean())
    rank = (theirs @ gold.T).argsort(dim=1, descending=True)
    correct = (rank == torch.arange(len(gold), device=device)[:, None])
    return aligned, drift, float(correct[:, 0].float().mean()), \
        float(correct[:, :5].any(dim=1).float().mean())


baseline = evaluate()
print(f"\nbefore: cosine {baseline[0]:.4f}   English kept {baseline[1]:.4f}   "
      f"R@1 {baseline[2]:.1%}   R@5 {baseline[3]:.1%}", flush=True)

optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
order = np.arange(len(english_train))
best, history = None, []

for epoch in range(1, args.epochs + 1):
    np.random.shuffle(order)
    total = 0.0
    for start in range(0, len(order), args.batch_size):
        idx = order[start:start + args.batch_size]
        gold = targets["train"][idx].to(device)
        predicted = F.normalize(
            embed([target_train[i] for i in idx], grad=True), dim=-1)
        loss = (1 - (predicted * gold).sum(-1)).mean()
        if args.keep_english:
            kept = F.normalize(
                embed([english_train[i] for i in idx], grad=True), dim=-1)
            loss = loss + args.keep_english * (1 - (kept * gold).sum(-1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss) * len(idx)

    aligned, drift, r1, r5 = evaluate()
    history.append({"epoch": epoch, "loss": total / len(order), "cosine": aligned,
                    "english_kept": drift, "recall@1": r1, "recall@5": r5})
    flag = ""
    if best is None or aligned > best:
        best, flag = aligned, "  <-"
        os.makedirs(rd.CHECKPOINT_DIR, exist_ok=True)
        torch.save({"text_state": {name: p.detach().cpu().clone()
                                   for name, p in model.named_parameters()
                                   if name in TEXT_ONLY},
                    "language": args.language,
                    "baseline": {"cosine": baseline[0], "english_kept": baseline[1],
                                 "recall@1": baseline[2], "recall@5": baseline[3]},
                    "best_cosine": best, "history": history, "args": vars(args)}, OUT)
    print(f"epoch {epoch:>2}/{args.epochs}  loss={total / len(order):.4f}  "
          f"cosine={aligned:.4f}  English kept={drift:.4f}  "
          f"R@1={r1:.1%}  R@5={r5:.1%}{flag}", flush=True)

print(f"\ncosine: {baseline[0]:.4f} -> {best:.4f}")
print(f"saved {os.path.relpath(OUT, rd.languages.REPO)}")
