import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "dataset")
EMBED_DIR = os.path.join(DATA_DIR, "clip_embeddings")

TARGETS = {"glide": "hbpp_glide", "sdxl": "hbpp_sdxl"}
SPLIT_FILES = {
    "train": "pqpp_multilingual_train.csv",
    "val": "pqpp_multilingual_val.csv",
    "test": "pqpp_multilingual_test.csv",
}

PARAM_GRID = {"learning_rate": [1e-5, 1e-4, 5e-5], "weight_decay": [0, 0.1, 0.01]}
NUM_EPOCHS = 25
BATCH_SIZE = 128
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--target", required=True, choices=sorted(TARGETS))
parser.add_argument(
    "--matrix", default="dims", choices=["dims", "images", "text-images", "multiling"],
    help="'dims' reproduces the original code: np.corrcoef(embeddings.T), i.e. correlations between the 512 DIMENSIONS, each estimated from only 4 observations, giving a 512x512 matrix of rank at most 3. 'images' implements what the paper DESCRIBES: cosine between every pair of IMAGES, a 4x4 matrix on generation or 25x25 on retrieval, well conditioned. 'text-images' adds the prompt as a fifth element, so the matrix becomes 5x5 and its last row holds text-image similarity, the native CLIP signal that the original method discards. 'multiling' places two versions of the prompt as separate elements, giving 6x6, so disagreement between languages becomes a feature: if a translation aligns differently with the images, the prompt is probably ambiguous.")
parser.add_argument(
    "--embed-tag", default="longclip-b",
    help="encoderul de imagine. 'longclip-b' = cel din paper; "
    "'xlmr-vitb32' = multilingv neadaptat; "
    "'xlmr-vitb32-ft-romanian_reviewed' = multilingv adaptat pe romana. "
    "Predictorul nu vede textul, dar vede embeddings, deci un encoder adaptat "
    "pe alta limba e singura cale prin care limba il poate influenta.")
args = parser.parse_args()

TAG = args.embed_tag
TARGET_COLUMN = TARGETS[args.target]
RUN = f"{args.target}__{TAG}" + {"images": "__imgmat", "text-images": "__txtmat",
                                 "multiling": "__mlmat"}.get(args.matrix, "")
RESULTS_DIR = os.path.join(HERE, "results", "corrcnn")
PREDICTIONS_DIR = os.path.join(HERE, "predictions", "corrcnn")
for directory in [RESULTS_DIR, PREDICTIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MATRIX_SIZE = {"images": 4, "text-images": 5, "multiling": 6}.get(args.matrix, 512)
TEXT_ITEMS = {"text-images": 1, "multiling": 2}.get(args.matrix, 0)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

image_npz = np.load(os.path.join(EMBED_DIR, f"image_embeddings_{TAG}.npz"), allow_pickle=True)

text_npz = None
if TEXT_ITEMS:
    text_path = os.path.join(EMBED_DIR, f"text_embeddings_{TAG}.npz")
    assert os.path.exists(text_path), f"lipseste {os.path.basename(text_path)}"
    text_npz = np.load(text_path, allow_pickle=True)
    TEXT_COLUMNS = (["text_english"] if args.matrix == "text-images"
                    else ["text_english", "text_romanian_reviewed"])
frames = []
for split_name, filename in SPLIT_FILES.items():
    frame = pd.read_csv(os.path.join(DATA_DIR, filename))
    frame["split"] = split_name
    frames.append(frame)
prompts = pd.concat(frames, ignore_index=True)
prompts["prompt_key"] = prompts["source"] + ":" + prompts["caption_id"].astype(str)

order = pd.DataFrame(
    {"prompt_index": image_npz["prompt_index"], "row": np.arange(len(image_npz["prompt_index"]))}
)
rows_by_prompt = np.full((len(prompts), 4), -1, dtype=np.int64)
for prompt_index, group in order.groupby("prompt_index", sort=True):
    rows_by_prompt[prompt_index] = group["row"].to_numpy()
assert (rows_by_prompt >= 0).all()

grouped = torch.from_numpy(image_npz["embeddings"][rows_by_prompt.reshape(-1)]).view(
    len(prompts), 4, -1
)
if TEXT_ITEMS:
    assert (text_npz["prompt_key"].astype(str) == prompts["prompt_key"].to_numpy()).all(),\
        'text embeddings are not aligned with the split'
    texts = torch.from_numpy(
        np.stack([text_npz[c] for c in TEXT_COLUMNS], axis=1).astype(np.float32))
    grouped = torch.cat([texts, grouped], dim=1)
    print(f"  elemente per prompt: {TEXT_ITEMS} text + 4 imagini = {grouped.shape[1]}")
prompts["normalized_target"] = (prompts[TARGET_COLUMN] + 1) / 3

print(f"run={RUN}  (correlation CNN, fara text)")
print(f"  embeddings: {tuple(grouped.shape)}")

def correlation_matrices(batch):
    if TEXT_ITEMS or args.matrix == "images":

        unit = batch / batch.norm(dim=2, keepdim=True).clamp_min(1e-8)
        return unit @ unit.transpose(1, 2)

    x = batch.transpose(1, 2)
    centered = x - x.mean(dim=2, keepdim=True)

    norm = centered.norm(dim=2, keepdim=True).clamp_min(1e-8)
    unit = centered / norm
    return unit @ unit.transpose(1, 2)

def split_tensors(split):
    mask = (prompts["split"] == split).to_numpy()
    positions = np.flatnonzero(mask)
    return (
        grouped[positions],
        torch.from_numpy(prompts["normalized_target"].to_numpy()[positions]).float(),
        positions,
    )

train_emb, train_y, _ = split_tensors("train")
val_emb, val_y, _ = split_tensors("val")
test_emb, test_y, test_positions = split_tensors("test")
train_y, val_y = train_y.to(device), val_y.to(device)
print(f"  train={len(train_emb)}  val={len(val_emb)}  test={len(test_emb)}")

class CNNRegressor(nn.Module):

    def __init__(self):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) if args.matrix == "dims" else nn.Identity()
        side = MATRIX_SIZE // 8 if args.matrix == "dims" else MATRIX_SIZE
        self.fc1 = nn.Linear(64 * side * side, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.sigmoid(self.fc2(x))

def predict(model, embeddings, batch_size=64):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            chunk = embeddings[start : start + batch_size].to(device)
            outputs.append(model(correlation_matrices(chunk)).squeeze(1))
    return torch.cat(outputs)

def train_one_config(lr, decay):
    set_seed(SEED)
    model = CNNRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loss_fn = nn.MSELoss()
    best = {"val_mse": float("inf")}

    for epoch in range(NUM_EPOCHS):
        model.train()
        permutation = torch.randperm(len(train_emb))
        for start in range(0, len(permutation), BATCH_SIZE):
            batch = permutation[start : start + BATCH_SIZE]
            inputs = correlation_matrices(train_emb[batch].to(device))
            loss = loss_fn(model(inputs).squeeze(1), train_y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_mse = float(((predict(model, val_emb) - val_y) ** 2).mean())
        if val_mse < best["val_mse"]:
            best = {
                "learning_rate": lr, "weight_decay": decay, "epoch": epoch + 1,
                "val_mse": val_mse,
                "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
            }
    del model
    torch.cuda.empty_cache()
    return best

best_overall = {"val_mse": float("inf")}
grid_results = []
for lr in PARAM_GRID["learning_rate"]:
    for decay in PARAM_GRID["weight_decay"]:
        config = train_one_config(lr, decay)
        grid_results.append({k: v for k, v in config.items() if k != "state_dict"})
        print(f"  lr={lr:g} wd={decay:g}  epoca {config['epoch']:>2}  "
              f"val_MSE={config['val_mse']:.5f}", flush=True)
        if config["val_mse"] < best_overall["val_mse"]:
            best_overall = config

best_config = {k: v for k, v in best_overall.items() if k != "state_dict"}
print(f"\nBest: lr={best_config['learning_rate']:g} wd={best_config['weight_decay']:g} "
      f"epoca {best_config['epoch']}  val_MSE={best_config['val_mse']:.5f}")

model = CNNRegressor().to(device)
model.load_state_dict(best_overall["state_dict"])

predictions = predict(model, test_emb).cpu().numpy()
predictions_raw = predictions * 3 - 1
test_prompts = prompts.iloc[test_positions].reset_index(drop=True)
true_raw = test_prompts[TARGET_COLUMN].to_numpy()
true_norm = test_y.numpy()

def metrics_for(true_r, pred_r, true_n, pred_n):
    pearson, pearson_p = scipy.stats.pearsonr(true_r, pred_r)
    kendall, kendall_p = scipy.stats.kendalltau(true_r, pred_r)
    spearman, spearman_p = scipy.stats.spearmanr(true_r, pred_r)
    return {
        "n": int(len(true_r)),
        "pearson": float(pearson), "pearson_p": float(pearson_p),
        "kendall": float(kendall), "kendall_p": float(kendall_p),
        "spearman": float(spearman), "spearman_p": float(spearman_p),
        "mse": float(mean_squared_error(true_n, pred_n)),
        "rmse": float(np.sqrt(mean_squared_error(true_n, pred_n))),
        "mae": float(np.mean(np.abs(true_n - pred_n))),
        "r2": float(r2_score(true_n, pred_n)),
        "bias_raw": float(np.mean(pred_r - true_r)),
    }

entry = {"total": metrics_for(true_raw, predictions_raw, true_norm, predictions)}
for source in test_prompts["source"].unique():
    mask = (test_prompts["source"] == source).to_numpy()
    if mask.sum() > 2:
        entry[source] = metrics_for(
            true_raw[mask], predictions_raw[mask], true_norm[mask], predictions[mask]
        )

results = {
    "run": RUN, "predictor": "correlation_cnn", "target": args.target,
    "target_column": TARGET_COLUMN, "target_family": "generative",
    "image_encoder": TAG,
    "language_independent": True,
    "note": 'the predictor does not use the prompt text; the result is identical for any language',
    "grid_complete": len(grid_results) == 9,
    "param_grid": PARAM_GRID, "num_epochs": NUM_EPOCHS,
    "best_config": best_config,
    "evaluation": entry,
}

total = entry["total"]
print(f"\n=== test ===")
print(f"  Pearson={total['pearson']:.3f} (p={total['pearson_p']:.2e})  "
      f"Kendall={total['kendall']:.3f}  R2={total['r2']:.4f}")
for source in test_prompts["source"].unique():
    if source in entry:
        s = entry[source]
        print(f"    {source:<10} n={s['n']:<5} Pearson={s['pearson']:.3f} "
              f"Kendall={s['kendall']:.3f}")

pd.DataFrame({
    "caption_id": test_prompts["caption_id"].to_numpy(),
    "source": test_prompts["source"].to_numpy(),
    "true_score": true_raw, "predicted_score": predictions_raw,
}).to_csv(os.path.join(PREDICTIONS_DIR, f"{RUN}.csv"), index=False)

with open(os.path.join(RESULTS_DIR, f"{RUN}.json"), "w") as handle:
    json.dump(results, handle, indent=2, ensure_ascii=False)
print(f"\nrezultate -> results/corrcnn/{RUN}.json")
