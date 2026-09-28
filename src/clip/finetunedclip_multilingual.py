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
DEFAULT_TAG = "xlmr-vitb32"

ALL_LANGUAGES = ["english", "romanian", "romanian_reviewed"]

TARGETS = {"glide": "hbpp_glide", "sdxl": "hbpp_sdxl"}

SPLIT_FILES = {
    "train": "pqpp_multilingual_train.csv",
    "val": "pqpp_multilingual_val.csv",
    "test": "pqpp_multilingual_test.csv",
}

PARAM_GRID = {"learning_rate": [1e-5, 1e-4, 5e-5], "weight_decay": [0, 0.1, 0.01]}
NUM_EPOCHS = 100
BATCH_SIZE = 256
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--language", required=True, choices=ALL_LANGUAGES)
parser.add_argument(
    "--embed-tag",
    default=DEFAULT_TAG,
    help="which embedding set to use; multilingual by default, 'longclip-b' being the monolingual control model from the paper",
)
parser.add_argument("--target", required=True, choices=sorted(TARGETS))
parser.add_argument(
    "--normalize",
    default="none",
    choices=["l2", "none"],
    help="'none' (default) concatenates raw embeddings, as in the original code. The image norm is ~17.3 and the text norm ~0.53, a 33x ratio that looks like it would unbalance the input, but measurement shows the opposite: on GLIDE in the pivot language 'none' gives Pearson 0.663 against 0.644 with 'l2'. The vector norms appear to carry signal that normalization erases.",
)
parser.add_argument(
    "--images",
    default="all",
    choices=["all", "target"],
    help="'all' uses all 4 images for any target, as in the original code; 'target' only the 2 from the predicted model",
)
args = parser.parse_args()

TAG = args.embed_tag
TARGET_COLUMN = TARGETS[args.target]
RUN = f"{args.target}__{args.language}"

suffix = "clip" if TAG == DEFAULT_TAG else f"clip_{TAG}"
RESULTS_DIR = os.path.join(HERE, "results", suffix)
PREDICTIONS_DIR = os.path.join(HERE, "predictions", suffix)
for directory in [RESULTS_DIR, PREDICTIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

text_npz = np.load(os.path.join(EMBED_DIR, f"text_embeddings_{TAG}.npz"), allow_pickle=True)
image_npz = np.load(os.path.join(EMBED_DIR, f"image_embeddings_{TAG}.npz"), allow_pickle=True)

frames = []
for split_name, filename in SPLIT_FILES.items():
    frame = pd.read_csv(os.path.join(DATA_DIR, filename))
    frame["split"] = split_name
    frames.append(frame)
prompts = pd.concat(frames, ignore_index=True)
prompts["prompt_key"] = prompts["source"] + ":" + prompts["caption_id"].astype(str)

assert (text_npz["prompt_key"].astype(str) == prompts["prompt_key"].to_numpy()).all()
image_index = pd.DataFrame(
    {
        "prompt_index": image_npz["prompt_index"],
        "generator": image_npz["generator"].astype(str),
        "row": np.arange(len(image_npz["prompt_index"])),
    }
)
assert image_index.groupby("prompt_index").size().eq(4).all()

image_features = image_npz["embeddings"]

EMBED_DIM = image_features.shape[1]
INPUT_DIM = 2 * EMBED_DIM

LANGUAGES = [lang for lang in ALL_LANGUAGES if f"text_{lang}" in text_npz.files]
assert args.language in LANGUAGES, (
    f"embeddings-urile '{TAG}' nu contin limba {args.language}; disponibile: {LANGUAGES}"
)
text_features = {language: text_npz[f"text_{language}"] for language in LANGUAGES}

if args.normalize == "l2":
    image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
    text_features = {
        language: features / np.linalg.norm(features, axis=1, keepdims=True)
        for language, features in text_features.items()
    }

rows_by_prompt = np.full((len(prompts), 4), -1, dtype=np.int64)
generator_by_slot = []
for prompt_index, group in image_index.groupby("prompt_index", sort=True):
    rows_by_prompt[prompt_index] = group["row"].to_numpy()
    generator_by_slot = list(group["generator"])
assert (rows_by_prompt >= 0).all()

if args.images == "target":
    keep = [slot for slot, generator in enumerate(generator_by_slot)
            if generator == args.target]
    assert len(keep) == 2, f"asteptam 2 imagini pentru {args.target}"
else:
    keep = list(range(4))
IMAGES_PER_PROMPT = len(keep)

prompts["normalized_target"] = (prompts[TARGET_COLUMN] + 1) / 3
assert prompts["normalized_target"].between(0, 1).all()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build(split, language):
    mask = (prompts["split"] == split).to_numpy()
    positions = np.flatnonzero(mask)
    text = text_features[language][positions]
    rows = rows_by_prompt[positions][:, keep]
    images = image_features[rows.reshape(-1)]
    text_repeated = np.repeat(text, IMAGES_PER_PROMPT, axis=0)
    features = np.hstack([text_repeated, images]).astype(np.float32)
    labels = prompts["normalized_target"].to_numpy()[positions].astype(np.float32)
    return (
        torch.from_numpy(features).to(device),
        torch.from_numpy(np.repeat(labels, IMAGES_PER_PROMPT)).unsqueeze(1).to(device),
        torch.from_numpy(labels).to(device),
        positions,
    )

train_x, train_y, _, _ = build("train", args.language)
val_x, _, val_prompt_y, _ = build("val", args.language)

print(f"run={RUN}  (CLIP, {args.normalize}, imagini={args.images})")
print(f"  train={tuple(train_x.shape)}  val={tuple(val_x.shape)}")

class NeuralNetworkRegressor(nn.Module):
    def __init__(self):
        super(NeuralNetworkRegressor, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.sigmoid(self.fc3(x))

def predict_prompts(model, features):
    model.eval()
    with torch.no_grad():
        outputs = torch.cat(
            [model(features[i : i + 4096]) for i in range(0, len(features), 4096)]
        )
    return outputs.view(-1, IMAGES_PER_PROMPT).mean(dim=1)

def train_one_config(lr, decay):
    set_seed(SEED)
    model = NeuralNetworkRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loss_fn = nn.MSELoss()
    best = {"val_mse": float("inf")}

    for epoch in range(NUM_EPOCHS):
        model.train()
        order = torch.randperm(len(train_x), device=device)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            loss = loss_fn(model(train_x[batch]), train_y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        predictions = predict_prompts(model, val_x)
        val_mse = float(((predictions - val_prompt_y) ** 2).mean())
        if val_mse < best["val_mse"]:
            best = {
                "learning_rate": lr,
                "weight_decay": decay,
                "epoch": epoch + 1,
                "val_mse": val_mse,
                "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
            }
    return best

best_overall = {"val_mse": float("inf")}
grid_results = []
for lr in PARAM_GRID["learning_rate"]:
    for decay in PARAM_GRID["weight_decay"]:
        config = train_one_config(lr, decay)
        grid_results.append({k: v for k, v in config.items() if k != "state_dict"})
        print(f"  lr={lr:g} wd={decay:g}  epoca {config['epoch']:>3}  "
              f"val_MSE={config['val_mse']:.5f}")
        if config["val_mse"] < best_overall["val_mse"]:
            best_overall = config

best_config = {k: v for k, v in best_overall.items() if k != "state_dict"}
print(f"\nBest: lr={best_config['learning_rate']:g} wd={best_config['weight_decay']:g} "
      f"epoca {best_config['epoch']}  val_MSE={best_config['val_mse']:.5f}")

model = NeuralNetworkRegressor().to(device)
model.load_state_dict(best_overall["state_dict"])
torch.save(
    {"model_state_dict": best_overall["state_dict"], "config": best_config,
     "language": args.language, "target": args.target, "normalize": args.normalize,
     "images": args.images, "embed_tag": TAG},
    os.path.join(RESULTS_DIR, f"{RUN}.pth"),
)

test_mask = (prompts["split"] == "test").to_numpy()
test_prompts = prompts[test_mask].reset_index(drop=True)
true_raw = test_prompts[TARGET_COLUMN].to_numpy()
true_norm = test_prompts["normalized_target"].to_numpy()

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

results = {
    "run": RUN, "predictor": "finetuned_clip", "language": args.language,
    "target": args.target, "target_column": TARGET_COLUMN,
    "target_family": "generative", "model_name": TAG, "embed_dim": int(EMBED_DIM),
    "eval_languages": LANGUAGES,
    "normalize": args.normalize, "images": args.images,
    "images_per_prompt": IMAGES_PER_PROMPT,
    "grid_complete": len(grid_results) == 9,
    "param_grid": PARAM_GRID, "num_epochs": NUM_EPOCHS,
    "best_config": best_config, "evaluations": {},
}

for language in LANGUAGES:
    features, _, _, _ = build("test", language)
    predictions = predict_prompts(model, features).cpu().numpy()
    predictions_raw = predictions * 3 - 1

    entry = {
        "setting": 'in-language' if language == args.language else "transfer zero-shot",
        "total": metrics_for(true_raw, predictions_raw, true_norm, predictions),
    }
    for source in test_prompts["source"].unique():
        source_mask = (test_prompts["source"] == source).to_numpy()
        if source_mask.sum() > 2:
            entry[source] = metrics_for(
                true_raw[source_mask], predictions_raw[source_mask],
                true_norm[source_mask], predictions[source_mask],
            )
    results["evaluations"][language] = entry

    pd.DataFrame({
        "caption_id": test_prompts["caption_id"].to_numpy(),
        "source": test_prompts["source"].to_numpy(),
        "true_score": true_raw, "predicted_score": predictions_raw,
    }).to_csv(os.path.join(PREDICTIONS_DIR, f"{RUN}__on_{language}.csv"), index=False)

    total = entry["total"]
    print(f"\n=== test pe {language} ({entry['setting']}) ===")
    print(f"  Pearson={total['pearson']:.3f} (p={total['pearson_p']:.2e})  "
          f"Kendall={total['kendall']:.3f}  R2={total['r2']:.4f}")
    for source in test_prompts["source"].unique():
        if source in entry:
            s = entry[source]
            print(f"    {source:<10} n={s['n']:<5} Pearson={s['pearson']:.3f} "
                  f"Kendall={s['kendall']:.3f}")

with open(os.path.join(RESULTS_DIR, f"{RUN}.json"), "w") as handle:
    json.dump(results, handle, indent=2, ensure_ascii=False)
print(f"\nrezultate -> results/clip/{RUN}.json")
