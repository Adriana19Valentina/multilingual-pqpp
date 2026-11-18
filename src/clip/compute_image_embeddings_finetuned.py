import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import languages

DRAWBENCH_ID_OFFSET = 10000

parser = argparse.ArgumentParser()
parser.add_argument("--language", default=languages.TARGET_LANGUAGE,
                    choices=[l for l in languages.LANGUAGES if l != languages.PIVOT])
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--num-workers", type=int, default=12)
args = parser.parse_args()

TAG = f"longclip-b-ft-{args.language}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, languages.THIRD_PARTY)
from huggingface_hub import hf_hub_download
from longclip_model import longclip

checkpoint = torch.load(
    os.path.join(languages.CHECKPOINT_ROOT, "clip", f"clip_finetuned_{args.language}_longclip-b.pt"),
    map_location="cpu", weights_only=False)
model, preprocess = longclip.load(
    hf_hub_download("BeichenZhang/LongCLIP-B", "longclip-B.pt"), device=device)
model.float()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval().to(device)
print(f"adapted weights: epoch {checkpoint['epoch']}, "
      f"R@1 {checkpoint['baseline_recall@1']:.3f} -> {checkpoint['recall@1']:.3f}")

base = np.load(os.path.join(languages.EMBED_DIR, "image_embeddings_longclip-b.npz"), allow_pickle=True)
paths = [
    os.path.join(languages.IMAGES_DIR, str(int(caption_id)), str(image_file)) if source == "mscoco"
    else os.path.join(languages.DRAWBENCH_DIR, str(int(caption_id) + DRAWBENCH_ID_OFFSET), str(image_file))
    for caption_id, source, image_file in zip(base["caption_id"], base["source"], base["image_file"])
]
missing = [p for p in paths if not os.path.exists(p)]
assert not missing, f"{len(missing)} missing images, first: {missing[0]}"
print(f"{TAG}: {len(paths)} images to encode")


class ImageDataset(Dataset):
    def __len__(self):
        return len(paths)

    def __getitem__(self, idx):
        with Image.open(paths[idx]) as image:
            return preprocess(image.convert("RGB"))


loader = DataLoader(ImageDataset(), batch_size=args.batch_size,
                    num_workers=args.num_workers, pin_memory=True)
chunks = []
with torch.no_grad():
    for step, batch in enumerate(loader, start=1):
        chunks.append(model.encode_image(batch.to(device, non_blocking=True)).float().cpu().numpy())
        if step % 100 == 0 or step == len(loader):
            print(f"  {min(step * args.batch_size, len(paths))}/{len(paths)}", flush=True)
embeddings = np.concatenate(chunks).astype(np.float32)

before = base["embeddings"] / np.linalg.norm(base["embeddings"], axis=1, keepdims=True)
after = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
drift = (before * after).sum(1)
print(f"cosine to the original embeddings: mean {drift.mean():.4f}, min {drift.min():.4f}")

out = os.path.join(languages.EMBED_DIR, f"image_embeddings_{TAG}.npz")
np.savez_compressed(out, **{key: base[key] for key in base.files if key != "embeddings"},
                    embeddings=embeddings)
print(f"wrote: {os.path.basename(out)}  {embeddings.shape}")
