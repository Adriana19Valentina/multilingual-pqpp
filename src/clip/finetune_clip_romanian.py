import argparse
import json
import os
import random

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "dataset")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
OUT_DIR = os.path.join(HERE, "checkpoints")

ENCODERS = {
    "xlmr-vitb32": ("open_clip", "xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),
    "longclip-b": ("longclip", "BeichenZhang/LongCLIP-B", "longclip-B.pt"),
}
LANGUAGE_COLUMNS = {
    "english": "caption",
    "romanian": "caption_romanian",
    "romanian_reviewed": "caption_romanian_reviewed",
}

parser = argparse.ArgumentParser()
parser.add_argument("--language", default="romanian_reviewed", choices=sorted(LANGUAGE_COLUMNS))
parser.add_argument("--encoder", default="xlmr-vitb32", choices=sorted(ENCODERS))
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-6,
                    help="foarte mica: 6.080 de perechi sunt putine pentru CLIP, "
                    "iar o rata mare duce la uitare catastrofala")
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
column = LANGUAGE_COLUMNS[args.language]

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

KIND, MODEL_NAME, PRETRAINED = ENCODERS[args.encoder]
if KIND == "open_clip":
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
else:
    import sys as _sys
    _sys.path.insert(0, os.path.join(HERE, "..", "third_party"))
    from huggingface_hub import hf_hub_download
    from longclip_model import longclip

    model, preprocess = longclip.load(
        hf_hub_download(MODEL_NAME, PRETRAINED), device=device)
    tokenizer = longclip.tokenize

    model.float()
model.to(device)
print(f"encoder: {args.encoder} ({MODEL_NAME})")

class PairDataset(Dataset):

    def __init__(self, split):
        frame = pd.read_csv(os.path.join(DATA_DIR, f"pqpp_multilingual_{split}.csv"))

        frame = frame[frame["source"] == "mscoco"].reset_index(drop=True)
        self.texts = [str(x) for x in frame[column]]
        self.paths = [
            os.path.join(IMAGES_DIR, str(int(i)), "image_6.png")
            for i in frame["caption_id"]
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            return preprocess(image.convert("RGB")), self.texts[idx]

def collate(batch):
    images, texts = zip(*batch)
    return torch.stack(images), tokenizer(list(texts))

train_loader = DataLoader(PairDataset("train"), batch_size=args.batch_size, shuffle=True,
                          num_workers=8, pin_memory=True, collate_fn=collate, drop_last=True)
val_loader = DataLoader(PairDataset("val"), batch_size=args.batch_size, shuffle=False,
                        num_workers=8, pin_memory=True, collate_fn=collate)
print(f"limba={args.language}  train={len(train_loader.dataset)}  "
      f"val={len(val_loader.dataset)}  lr={args.lr:g}")

def contrastive_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = logit_scale * image_features @ text_features.T
    labels = torch.arange(len(logits), device=logits.device)

    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

@torch.no_grad()
def evaluate():
    model.eval()
    images, texts = [], []
    for batch_images, batch_texts in val_loader:
        images.append(F.normalize(model.encode_image(batch_images.to(device)).float(), dim=-1))
        texts.append(F.normalize(model.encode_text(batch_texts.to(device)).float(), dim=-1))
    images, texts = torch.cat(images), torch.cat(texts)
    similarity = texts @ images.T
    ranks = (similarity > similarity.diag().unsqueeze(1)).sum(1)
    return {
        "recall@1": float((ranks == 0).float().mean()),
        "recall@5": float((ranks < 5).float().mean()),
        "median_rank": float(ranks.float().median()) + 1,
    }

baseline = evaluate()
print(f"inainte de fine-tuning: R@1={baseline['recall@1']:.3f} "
      f"R@5={baseline['recall@5']:.3f} rang median={baseline['median_rank']:.0f}")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
history = [{"epoch": 0, **baseline}]
best = {"recall@1": baseline["recall@1"], "epoch": 0, "state_dict": None}

for epoch in range(1, args.epochs + 1):
    model.train()
    total, steps = 0.0, 0
    for batch_images, batch_texts in train_loader:
        image_features = model.encode_image(batch_images.to(device, non_blocking=True))
        text_features = model.encode_text(batch_texts.to(device, non_blocking=True))
        loss = contrastive_loss(image_features, text_features, model.logit_scale.exp())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.logit_scale.clamp_(0, np.log(100))
        total += float(loss)
        steps += 1

    metrics = evaluate()
    history.append({"epoch": epoch, "train_loss": total / steps, **metrics})
    print(f"epoca {epoch}  loss={total / steps:.4f}  R@1={metrics['recall@1']:.3f} "
          f"R@5={metrics['recall@5']:.3f} rang median={metrics['median_rank']:.0f}", flush=True)
    if metrics["recall@1"] > best["recall@1"]:
        best = {"recall@1": metrics["recall@1"], "epoch": epoch,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

if best["state_dict"] is None:
    print("\nATENTIE: nicio epoca nu a depasit modelul pre-antrenat pe validare.")
    print("Fine-tuningul degradeaza encoderul; nu se salveaza checkpoint.")
else:
    suffix = "" if args.encoder == "xlmr-vitb32" else f"_{args.encoder}"
    path = os.path.join(OUT_DIR, f"clip_finetuned_{args.language}{suffix}.pt")
    torch.save({"model_state_dict": best["state_dict"], "model_name": MODEL_NAME,
                "encoder": args.encoder, "kind": KIND,
                "pretrained": PRETRAINED, "language": args.language,
                "epoch": best["epoch"], "recall@1": best["recall@1"],
                "baseline_recall@1": baseline["recall@1"], "args": vars(args)}, path)
    print(f"\nsalvat: {os.path.relpath(path, HERE)}  (epoca {best['epoch']}, "
          f"R@1 {baseline['recall@1']:.3f} -> {best['recall@1']:.3f})")

_suffix = "" if args.encoder == "xlmr-vitb32" else f"_{args.encoder}"
with open(os.path.join(OUT_DIR,
                       f"clip_finetuned_{args.language}{_suffix}.log.json"), "w") as handle:
    json.dump({"args": vars(args), "history": history}, handle, indent=2)
