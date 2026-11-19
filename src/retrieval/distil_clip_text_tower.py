import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

import retrieval_data as rd

MODEL_ID = "openai/clip-vit-base-patch32"

parser = argparse.ArgumentParser(
    description="Distil the target language into CLIP's English text tower. "
                "The vision tower stays frozen, so the cached corpus embeddings "
                "remain valid.")
parser.add_argument("--language", default=rd.languages.TARGET_LANGUAGE,
                    choices=rd.target_languages())
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--keep-english", type=float, default=0.5)
args = parser.parse_args()
OUT = rd.checkpoint_path("clip-vitb32", "distilled", args.language)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher = CLIPModel.from_pretrained(MODEL_ID).eval().to(device)
student = CLIPModel.from_pretrained(MODEL_ID).to(device)
tokenizer = CLIPProcessor.from_pretrained(MODEL_ID).tokenizer

for p in teacher.parameters():
    p.requires_grad = False
for name, p in student.named_parameters():
    p.requires_grad = name.startswith(("text_model", "text_projection"))
trainable = [p for p in student.parameters() if p.requires_grad]
print(f"{MODEL_ID} on {device}")
print(f"trainable: {sum(p.numel() for p in trainable) / 1e6:.1f} M "
      f"of {sum(p.numel() for p in student.parameters()) / 1e6:.1f} M")

data = {}
for split in ["train", "val"]:
    data[split] = rd.parallel_captions(args.language, split)
    print(f"{split}: {len(data[split][0])} parallel captions")


def embed(model, texts, batch=256, grad=False):
    out = []
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        for start in range(0, len(texts), batch):
            tokens = tokenizer(texts[start:start + batch], return_tensors="pt",
                               padding=True, truncation=True, max_length=77).to(device)
            out.append(model.get_text_features(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"]).float())
    return torch.cat(out)


with torch.no_grad():
    targets = {split: F.normalize(embed(teacher, english), dim=-1).cpu()
               for split, (english, _) in data.items()}


def evaluate():
    student.eval()
    english, target_lang = data["val"]
    with torch.no_grad():
        gold = targets["val"].to(device)
        theirs = F.normalize(embed(student, target_lang), dim=-1)
        ours_en = F.normalize(embed(student, english), dim=-1)
    aligned = float((theirs * gold).sum(-1).mean())
    drift = float((ours_en * gold).sum(-1).mean())
    rank = (theirs @ gold.T).argsort(dim=1, descending=True)
    correct = (rank == torch.arange(len(gold), device=device)[:, None])
    student.train()
    return aligned, drift, float(correct[:, 0].float().mean()), \
        float(correct[:, :5].any(dim=1).float().mean())


baseline = evaluate()
print(f"\nbefore: cosine {baseline[0]:.4f}   English kept {baseline[1]:.4f}   "
      f"R@1 {baseline[2]:.1%}   R@5 {baseline[3]:.1%}")

optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
english_train, target_train = data["train"]
order = np.arange(len(english_train))
best, history = None, []

for epoch in range(1, args.epochs + 1):
    np.random.shuffle(order)
    total = 0.0
    for start in range(0, len(order), args.batch_size):
        idx = order[start:start + args.batch_size]
        gold = targets["train"][idx].to(device)
        predicted = F.normalize(
            embed(student, [target_train[i] for i in idx], grad=True), dim=-1)
        loss = (1 - (predicted * gold).sum(-1)).mean()
        if args.keep_english:
            kept = F.normalize(
                embed(student, [english_train[i] for i in idx], grad=True), dim=-1)
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
        torch.save({"model_state_dict": {k: v.detach().cpu().clone()
                                         for k, v in student.state_dict().items()},
                    "model_id": MODEL_ID, "language": args.language,
                    "baseline": {"cosine": baseline[0], "english_kept": baseline[1],
                                 "recall@1": baseline[2], "recall@5": baseline[3]},
                    "best_cosine": best, "history": history, "args": vars(args)}, OUT)
    print(f"epoch {epoch:>2}/{args.epochs}  loss={total / len(order):.4f}  "
          f"cosine={aligned:.4f}  English kept={drift:.4f}  "
          f"R@1={r1:.1%}  R@5={r5:.1%}{flag}", flush=True)

print(f"\ncosine: {baseline[0]:.4f} -> {best:.4f}")
print(f"saved {os.path.relpath(OUT, rd.languages.REPO)}")
