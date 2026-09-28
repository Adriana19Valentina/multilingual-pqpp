import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "third_party"))
from longclip_model import longclip

IMAGES_DIR = os.path.join(HERE, "all")
OUT_DIR = os.path.join(HERE, "clip_embeddings")
TEST_IDS_FILE = os.path.join(OUT_DIR, "test2017_ids.txt")
TAG = "longclip-l"
HF_REPO, HF_FILE = "BeichenZhang/LongCLIP-L", "longclip-L.pt"

BATCH_SIZE = 64
NUM_WORKERS = 12

os.makedirs(OUT_DIR, exist_ok=True)

with open(TEST_IDS_FILE) as handle:
    test_ids = {int(line) for line in handle if line.strip()}

entries = []
for filename in os.listdir(IMAGES_DIR):
    match = re.match(r"0*(\d+)\.jpg$", filename)
    if match and int(match.group(1)) not in test_ids:
        entries.append((int(match.group(1)), os.path.join(IMAGES_DIR, filename)))
entries.sort()
image_ids = np.array([i for i, _ in entries], dtype=np.int64)
paths = [p for _, p in entries]
print(f"corpus: {len(paths)} images (train2017)")
assert len(paths) == 118287, f"expected 118287, found {len(paths)}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = hf_hub_download(HF_REPO, HF_FILE)
model, preprocess = longclip.load(checkpoint, device=device)
model.eval()
print(f"Long-CLIP-L: patch {model.visual.conv1.kernel_size}, "
      f"dim {model.text_projection.shape[1]}")

class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            return preprocess(image.convert("RGB"))

loader = DataLoader(
    ImageDataset(paths), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)

chunks = []
with torch.no_grad():
    for step, batch in enumerate(loader, start=1):
        features = model.encode_image(batch.to(device, non_blocking=True))
        chunks.append(features.float().cpu().numpy())
        if step % 100 == 0 or step == len(loader):
            done = min(step * BATCH_SIZE, len(paths))
            print(f"  {done}/{len(paths)} ({100 * done / len(paths):.0f}%)", flush=True)

embeddings = np.concatenate(chunks).astype(np.float32)
assert embeddings.shape[0] == len(paths)
print(f"\nembeddings: {embeddings.shape}  "
      f"mean norm {np.linalg.norm(embeddings, axis=1).mean():.2f}")

npz_path = os.path.join(OUT_DIR, f"corpus_embeddings_{TAG}.npz")
np.savez(npz_path, embeddings=embeddings, image_id=image_ids)

with open(os.path.join(OUT_DIR, f"corpus_embeddings_{TAG}.manifest.json"), "w") as handle:
    json.dump(
        {
            "created": datetime.now().isoformat(timespec="seconds"),
            "role": 'retrieval corpus for reconstructing the missing lists',
            "model": {
                "name": "Long-CLIP-L",
                "checkpoint": f"{HF_REPO}/{HF_FILE}",
                "embed_dim": int(embeddings.shape[1]),
                "normalized": False,
                "identified_by": "the query embeddings published in the repo "
                "(768-d) are reproduced by this model at cosine 1.0000",
            },
            "corpus": {
                "source": "MS COCO train2017",
                "images": int(len(paths)),
                "note": 'test2017 excluded: it contains no relevant image',
            },
        },
        handle, indent=2, ensure_ascii=False,
    )
print(f"wrote: {os.path.relpath(npz_path, HERE)} "
      f"({os.path.getsize(npz_path) / 2**20:.0f} MiB)")
