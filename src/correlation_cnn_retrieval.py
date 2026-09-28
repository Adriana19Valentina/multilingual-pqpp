import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "dataset")
EMBED_DIR = os.path.join(DATA_DIR, "clip_embeddings")
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GT_DIR = os.path.join(ROOT, "dataset", "retrieval", "ground_truth")

TARGETS = {
    "clip_p10": ("clip", "precision"),
    "clip_rr": ("clip", "reciprocal_rank"),
    "blip2_p10": ("blip2", "precision"),
    "blip2_rr": ("blip2", "reciprocal_rank"),
}
SPLIT_FILES = {
    "train": "pqpp_multilingual_train.csv",
    "val": "pqpp_multilingual_val.csv",
    "test": "pqpp_multilingual_test.csv",
}
TOP_K = 25
PARAM_GRID = {"learning_rate": [1e-5, 1e-4, 5e-5], "weight_decay": [0, 0.1, 0.01]}
NUM_EPOCHS = 25
BATCH_SIZE = 32
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--target", required=True, choices=sorted(TARGETS))
parser.add_argument("--encoder", default="longclip-b", choices=["longclip-b", "xlmr-vitb32"])
args = parser.parse_args()

SYSTEM, METRIC = TARGETS[args.target]
RUN = f"{args.target}__{args.encoder}"
RESULTS_DIR = os.path.join(HERE, "results", "corrcnn_retrieval")
PREDICTIONS_DIR = os.path.join(HERE, "predictions", "corrcnn_retrieval")
for directory in [RESULTS_DIR, PREDICTIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

image_npz = np.load(os.path.join(EMBED_DIR, f"retrieved_image_embeddings_{args.encoder}.npz"))
row_of_image = {int(i): r for r, i in enumerate(image_npz["image_id"])}
image_features = torch.from_numpy(image_npz["embeddings"])
retrieval_lists = np.load(os.path.join(EMBED_DIR, f"retrieval_lists_{SYSTEM}.npz"))

frames, targets, grouped = {}, {}, {}
for split, filename in SPLIT_FILES.items():
    frames[split] = pd.read_csv(os.path.join(DATA_DIR, filename))
    reference = pd.read_csv(
        os.path.join(GT_DIR, SYSTEM, f"{SYSTEM}_retrieval_{split}_results.csv")
    )
    targets[split] = torch.from_numpy(reference[METRIC].to_numpy()).float()
    rows = np.array([
        [row_of_image[int(i)] for i in query_row]
        for query_row in retrieval_lists[f"{split}_top{TOP_K}"]
    ])
    grouped[split] = image_features[rows.reshape(-1)].view(len(rows), TOP_K, -1)

print(f"run={RUN}  sistem={SYSTEM}  metrica={METRIC}  (fara text)")
print(f"  train={tuple(grouped['train'].shape)}  tinta medie={targets['train'].mean():.4f}")

def correlation_matrices(batch):
    x = batch.transpose(1, 2)
    centered = x - x.mean(dim=2, keepdim=True)
    unit = centered / centered.norm(dim=2, keepdim=True).clamp_min(1e-8)
    return unit @ unit.transpose(1, 2)

class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

def predict(model, embeddings, batch_size=32):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            chunk = embeddings[start : start + batch_size].to(device)
            outputs.append(model(correlation_matrices(chunk)).squeeze(1).cpu())
    return torch.cat(outputs)

def train_one_config(lr, decay):
    set_seed(SEED)
    model = CNNRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loss_fn = nn.MSELoss()
    best = {"val_mse": float("inf")}
    train_y = targets["train"].to(device)
    for epoch in range(NUM_EPOCHS):
        model.train()
        order = torch.randperm(len(grouped["train"]))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            inputs = correlation_matrices(grouped["train"][batch].to(device))
            loss = loss_fn(model(inputs).squeeze(1), train_y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_mse = float(((predict(model, grouped["val"]) - targets["val"]) ** 2).mean())
        if val_mse < best["val_mse"]:
            best = {"learning_rate": lr, "weight_decay": decay, "epoch": epoch + 1,
                    "val_mse": val_mse,
                    "state_dict": {k: v.clone() for k, v in model.state_dict().items()}}
    del model
    torch.cuda.empty_cache()
    return best

best_overall = {"val_mse": float("inf")}
grid = []
for lr in PARAM_GRID["learning_rate"]:
    for decay in PARAM_GRID["weight_decay"]:
        config = train_one_config(lr, decay)
        grid.append({k: v for k, v in config.items() if k != "state_dict"})
        print(f"  lr={lr:g} wd={decay:g}  epoca {config['epoch']:>2}  "
              f"val_MSE={config['val_mse']:.5f}", flush=True)
        if config["val_mse"] < best_overall["val_mse"]:
            best_overall = config

best_config = {k: v for k, v in best_overall.items() if k != "state_dict"}
print(f"\nBest: lr={best_config['learning_rate']:g} wd={best_config['weight_decay']:g} "
      f"epoca {best_config['epoch']}  val_MSE={best_config['val_mse']:.5f}")

model = CNNRegressor().to(device)
model.load_state_dict(best_overall["state_dict"])
predictions = predict(model, grouped["test"]).numpy()
expected = targets["test"].numpy()
test_frame = frames["test"]

entry = {}
for subset, mask in [("total", np.ones(len(test_frame), bool)),
                     ("mscoco", (test_frame["source"] == "mscoco").to_numpy()),
                     ("drawbench", (test_frame["source"] == "drawbench").to_numpy())]:
    if mask.sum() < 3:
        continue
    a, b = predictions[mask], expected[mask]
    if a.std() == 0 or b.std() == 0:
        entry[subset] = {"n": int(mask.sum()), "degenerate": True,
                         "pearson": 0.0, "pearson_p": 1.0, "kendall": 0.0, "kendall_p": 1.0}
        continue
    pearson, pearson_p = scipy.stats.pearsonr(b, a)
    kendall, kendall_p = scipy.stats.kendalltau(b, a)
    entry[subset] = {
        "n": int(mask.sum()), "pearson": float(pearson), "pearson_p": float(pearson_p),
        "kendall": float(kendall), "kendall_p": float(kendall_p),
        "mean_predicted": float(a.mean()), "mean_target": float(b.mean()),
    }

total = entry["total"]
print(f"\n=== test ===")
print(f"  Pearson={total['pearson']:+.3f} (p={total['pearson_p']:.1e})  "
      f"Kendall={total['kendall']:+.3f}   medii {total.get('mean_predicted', 0):.3f} "
      f"vs {total.get('mean_target', 0):.3f}")

pd.DataFrame({
    "caption_id": test_frame["caption_id"].to_numpy(),
    "source": test_frame["source"].to_numpy(),
    "true_score": expected, "predicted_score": predictions,
}).to_csv(os.path.join(PREDICTIONS_DIR, f"{RUN}.csv"), index=False)

with open(os.path.join(RESULTS_DIR, f"{RUN}.json"), "w") as handle:
    json.dump({"run": RUN, "predictor": "correlation_cnn_retrieval",
               "target": args.target, "system": SYSTEM, "metric": METRIC,
               "image_encoder": args.encoder, "language_independent": True,
               "note": 'does not use the query text; identical result for any language',
               "grid_complete": len(grid) == 9, "param_grid": PARAM_GRID,
               "num_epochs": NUM_EPOCHS, "best_config": best_config,
               "evaluation": entry}, handle, indent=2, ensure_ascii=False)
print(f"\nrezultate -> results/corrcnn_retrieval/{RUN}.json")
