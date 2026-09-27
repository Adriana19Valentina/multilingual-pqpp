import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(HERE, "clip_embeddings")
IMAGES_DIR = os.path.join(HERE, "all")

parser = argparse.ArgumentParser()
parser.add_argument("--encoder", required=True,
                    choices=["longclip-b", "xlmr-vitb32", "xlmr-vith14",
                             "xlmr-vitb32-ft-romanian_reviewed"])
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--num-workers", type=int, default=12)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.encoder == "longclip-b":
    sys.path.insert(0, os.path.join(HERE, "..", "third_party"))
    from huggingface_hub import hf_hub_download
    from longclip_model import longclip

    model, preprocess = longclip.load(
        hf_hub_download("BeichenZhang/LongCLIP-B", "longclip-B.pt"), device=device
    )
    model.eval()
    encode = model.encode_image
else:
    import open_clip

    name, pretrained = {
        "xlmr-vitb32": ("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),
        "xlmr-vith14": ("xlm-roberta-large-ViT-H-14", "frozen_laion5b_s13b_b90k"),
        "xlmr-vitb32-ft-romanian_reviewed": ("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),
    }[args.encoder]
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    if "-ft-" in args.encoder:
        _lang = args.encoder.split("-ft-")[1]
        _ck = torch.load(os.path.join(HERE, "..", "models", "checkpoints",
                                      f"clip_finetuned_{_lang}.pt"),
                         map_location="cpu", weights_only=False)
        model.load_state_dict(_ck["model_state_dict"])
        print(f"greutati adaptate: R@1 {_ck['baseline_recall@1']:.3f} -> {_ck['recall@1']:.3f}")
    model.eval().to(device)
    encode = model.encode_image

with open(os.path.join(EMBED_DIR, "retrieved_image_ids.json")) as handle:
    image_ids = np.array(json.load(handle), dtype=np.int64)
paths = [os.path.join(IMAGES_DIR, f"{i:012d}.jpg") for i in image_ids]
missing = [p for p in paths[:100] if not os.path.exists(p)]
assert not missing, f"lipsesc imagini, prima: {missing[0]}"
print(f"{args.encoder}: {len(paths)} imagini de encodat")

class ImageDataset(Dataset):
    def __len__(self):
        return len(paths)

    def __getitem__(self, idx):
        with Image.open(paths[idx]) as image:
            return preprocess(image.convert("RGB"))

loader = DataLoader(
    ImageDataset(), batch_size=args.batch_size, num_workers=args.num_workers,
    pin_memory=True,
)
chunks = []
with torch.no_grad():
    for step, batch in enumerate(loader, start=1):
        chunks.append(encode(batch.to(device, non_blocking=True)).float().cpu().numpy())
        if step % 100 == 0 or step == len(loader):
            done = min(step * args.batch_size, len(paths))
            print(f"  {done}/{len(paths)}", flush=True)

embeddings = np.concatenate(chunks).astype(np.float32)
out = os.path.join(EMBED_DIR, f"retrieved_image_embeddings_{args.encoder}.npz")
np.savez(out, embeddings=embeddings, image_id=image_ids)
print(f"scris: {os.path.basename(out)}  {embeddings.shape}  "
      f"norma medie {np.linalg.norm(embeddings, axis=1).mean():.2f}")
