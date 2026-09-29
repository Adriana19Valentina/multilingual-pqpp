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
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import languages

DATA_DIR = languages.DATA_DIR
SPLIT_FILES = languages.SPLIT_FILES
LANGUAGE_COLUMNS = languages.COLUMNS
STUDY_LANGUAGES = languages.LANGUAGES

TARGETS = {
    "glide": ("hbpp_glide", "generative"),
    "sdxl": ("hbpp_sdxl", "generative"),
    "hbpp_average": ("hbpp_average", "generative"),
    "clip_p10": ("p10_clip", "retrieval"),
    "clip_rr": ("rr_clip", "retrieval"),
    "blip2_p10": ("p10_blip2", "retrieval"),
    "blip2_rr": ("rr_blip2", "retrieval"),
    "p10_average": ("p10_average", "retrieval"),
    "rr_average": ("rr_average", "retrieval"),
}

MODEL_NAME = "bert-base-multilingual-cased"
OUTPUT_ACTIVATION = "relu"

BATCH_SIZE = 256
MICRO_BATCH_SIZE = 128
GRAD_ACCUM = BATCH_SIZE // MICRO_BATCH_SIZE
EVAL_BATCH_SIZE = 256
NUM_WORKERS = 4
SEED = 42

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PARAM_GRID = {
    "learning_rate": [1e-5, 1e-4, 5e-5],
    "num_epochs": [15],
    "weight_decay": [0, 0.1, 0.01],
}

parser = argparse.ArgumentParser()
parser.add_argument("--language", required=True, choices=sorted(LANGUAGE_COLUMNS))
parser.add_argument(
    "--eval-languages",
    nargs="+",
    default=STUDY_LANGUAGES,
    choices=sorted(LANGUAGE_COLUMNS),
    help='languages the winning checkpoint is evaluated on',
)
parser.add_argument("--target", required=True, choices=sorted(TARGETS))
parser.add_argument(
    "--precision",
    default="fp32",
    choices=["fp32", "bf16"],
    help='bf16 is ~1.5x faster but changes training numerically; fp32 (default) keeps the published protocol',
)
parser.add_argument(
    "--keep-checkpoints",
    action="store_true",
    help='keep every grid checkpoint (~413 MiB each); by default they are deleted once the grid completes, leaving only the winner',
)
args = parser.parse_args()

eval_languages = list(dict.fromkeys([args.language] + args.eval_languages))

autocast = (
    torch.autocast("cuda", dtype=torch.bfloat16)
    if args.precision == "bf16"
    else torch.autocast("cuda", enabled=False)
)

TEXT_COLUMN = LANGUAGE_COLUMNS[args.language]
TARGET_COLUMN, TARGET_FAMILY = TARGETS[args.target]
RUN = f"{args.target}__{args.language}"

CHECKPOINT_DIR = os.path.join(languages.CHECKPOINT_ROOT, "bert", RUN)
RESULTS_DIR = languages.RESULTS_DIR
PREDICTIONS_DIR = languages.PREDICTIONS_DIR
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best.pth")
GRID_RESULTS_PATH = os.path.join(CHECKPOINT_DIR, "grid_results.json")

for directory in [CHECKPOINT_DIR, RESULTS_DIR, PREDICTIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

def normalize(values):
    return (values + 1) / 3 if TARGET_FAMILY == "generative" else values

def denormalize(values):
    return values * 3 - 1 if TARGET_FAMILY == "generative" else values

def checkpoint_path_for(lr, weight_decay):
    return os.path.join(CHECKPOINT_DIR, f"lr{lr:g}_wd{weight_decay:g}.pth")

def marker_path_for(lr, weight_decay):
    return os.path.join(CHECKPOINT_DIR, f"lr{lr:g}_wd{weight_decay:g}.done.json")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

splits = languages.load_splits()

print("coverage:")
for name in list(splits):
    mask = languages.usable_mask(splits[name])
    languages.report_coverage(name, mask)
    splits[name] = splits[name][mask].reset_index(drop=True)

for name, df in splits.items():
    needed = [TARGET_COLUMN] + (
        [TEXT_COLUMN]
        if name != "test"
        else [LANGUAGE_COLUMNS[language] for language in eval_languages]
    )
    for column in needed:
        if column not in df.columns:
            raise SystemExit(languages.missing_column_message(
                column, args.language, df))
        assert df[column].notna().all(), f"{name}: missing values in {column}"
    df["normalized_target"] = normalize(df[TARGET_COLUMN])
    assert df["normalized_target"].between(0, 1).all(), f"{name}: target outside [0, 1]"

train_data, eval_data, test_data = splits["train"], splits["val"], splits["test"]
print(f"run={RUN}")
print(f"  text={TEXT_COLUMN}  target={TARGET_COLUMN} ({TARGET_FAMILY})")
print(f"  train={len(train_data)}  val={len(eval_data)}  test={len(test_data)}")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class CaptionDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, column):
        self.encodings = tokenizer(list(dataframe[column]), truncation=True)
        self.labels = dataframe["normalized_target"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": float(self.labels[idx]),
        }

def collate(batch):
    labels = torch.tensor([item.pop("label") for item in batch]).unsqueeze(-1).float()
    encoded = tokenizer.pad(batch, return_tensors="pt")
    return encoded["input_ids"], encoded["attention_mask"], labels

train_loader = DataLoader(
    CaptionDataset(train_data, TEXT_COLUMN),
    batch_size=MICRO_BATCH_SIZE,
    shuffle=True,
    collate_fn=collate,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
)
eval_loader = DataLoader(
    CaptionDataset(eval_data, TEXT_COLUMN),
    batch_size=EVAL_BATCH_SIZE,
    collate_fn=collate,
    num_workers=NUM_WORKERS,
    persistent_workers=NUM_WORKERS > 0,
)

def test_loader_for(column):
    return DataLoader(
        CaptionDataset(test_data, column),
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
    )

class BertRegressor(nn.Module):
    def __init__(self, n_outputs=1):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.linear2 = nn.Linear(512, n_outputs)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        x = self.linear1(pooled_output)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        return self.relu(x) if OUTPUT_ACTIVATION == "relu" else self.sigmoid(x)

    def get_embeddings(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        return self.linear1(pooled_output)

def evaluate_model(model, data_loader):
    model.eval()
    predictions, labels_list = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            with autocast:
                outputs = model(input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.float().cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    predictions = np.array(predictions).flatten()
    labels_array = np.array(labels_list).flatten()
    return (
        mean_squared_error(labels_array, predictions),
        r2_score(labels_array, predictions),
        predictions,
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device={device}\n")

def train_one_config(lr, decay, epochs):
    set_seed(SEED)
    model = BertRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    config_best = {"val_mse": float("inf")}

    try:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pending = False
            for step, batch in enumerate(train_loader, start=1):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                with autocast:
                    outputs = model(input_ids, attention_mask)
                    loss = torch.nn.functional.mse_loss(outputs, labels)
                (loss / GRAD_ACCUM).backward()
                pending = True
                if step % GRAD_ACCUM == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    pending = False
            if pending:
                optimizer.step()
                optimizer.zero_grad()

            mse_eval, r2_eval, _ = evaluate_model(model, eval_loader)

            if mse_eval < config_best["val_mse"]:
                config_best = {
                    "learning_rate": lr,
                    "weight_decay": decay,
                    "epoch": epoch + 1,
                    "val_mse": float(mse_eval),
                    "val_r2": float(r2_eval),
                    "checkpoint": checkpoint_path_for(lr, decay),
                }
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config_best,
                        "text_column": TEXT_COLUMN,
                        "target_column": TARGET_COLUMN,
                        "target_family": TARGET_FAMILY,
                        "language": args.language,
                        "target": args.target,
                        "model_name": MODEL_NAME,
                        "output_activation": OUTPUT_ACTIVATION,
                        "seed": SEED,
                    },
                    config_best["checkpoint"],
                )

            print(
                f"  lr={lr:g} wd={decay:g} epoch={epoch + 1}/{epochs} "
                f"val_MSE={mse_eval:.5f} val_R2={r2_eval:.4f}"
            )
    finally:
        del model, optimizer
        torch.cuda.empty_cache()

    return config_best

best_mse = float("inf")
best_params = {}
grid_results = []

for lr in PARAM_GRID["learning_rate"]:
    for epochs in PARAM_GRID["num_epochs"]:
        for decay in PARAM_GRID["weight_decay"]:
            marker = marker_path_for(lr, decay)

            if os.path.exists(marker):
                with open(marker) as handle:
                    config_best = json.load(handle)
                print(
                    f"  lr={lr:g} wd={decay:g} deja rulat -> se refoloseste "
                    f"(epoch {config_best['epoch']}, val_MSE={config_best['val_mse']:.5f})"
                )
            else:
                try:
                    config_best = train_one_config(lr, decay, epochs)
                except torch.cuda.OutOfMemoryError:

                    print(f"  !! lr={lr:g} wd={decay:g} CUDA OOM, skipping")
                    torch.cuda.empty_cache()
                    continue

                with open(marker, "w") as handle:
                    json.dump(config_best, handle, indent=2)

            grid_results.append(config_best)
            if config_best["val_mse"] < best_mse:
                best_mse = config_best["val_mse"]
                best_params = config_best

expected = len(PARAM_GRID["learning_rate"]) * len(PARAM_GRID["weight_decay"])
complete = len(grid_results) == expected

if not best_params:
    raise SystemExit('No configuration completed successfully; nothing to evaluate.')
if not complete:
    print(
        f"\nWARNING: {expected - len(grid_results)}/{expected} configurations are "
        f"missing (OOM). Rerun to pick them up; until then the result is NOT "
        f"comparable with runs that completed the whole grid."
    )

with open(GRID_RESULTS_PATH, "w") as handle:
    json.dump(
        {"grid": grid_results, "best": best_params, "complete": complete},
        handle,
        indent=2,
        ensure_ascii=False,
    )

os.replace(best_params["checkpoint"], BEST_CHECKPOINT)
best_params["checkpoint"] = BEST_CHECKPOINT

if not args.keep_checkpoints and complete:

    removed = 0
    for filename in os.listdir(CHECKPOINT_DIR):
        if filename.endswith(".pth") and filename != "best.pth":
            os.remove(os.path.join(CHECKPOINT_DIR, filename))
            removed += 1
    print(f"  ({removed} non-winning checkpoints deleted; --keep-checkpoints preserves them)")

print(f"\nBest: lr={best_params['learning_rate']:g} wd={best_params['weight_decay']:g} "
      f"epoch {best_params['epoch']}  val_MSE={best_mse:.5f}")

model = BertRegressor().to(device)
model.load_state_dict(torch.load(BEST_CHECKPOINT)["model_state_dict"])

true_raw = test_data[TARGET_COLUMN].values
true_norm = test_data["normalized_target"].values

results = {
    "run": RUN,
    "language": args.language,
    "target": args.target,
    "target_column": TARGET_COLUMN,
    "target_family": TARGET_FAMILY,
    "text_column": TEXT_COLUMN,
    "model_name": MODEL_NAME,
    "grid_complete": complete,
    "param_grid": {k: v for k, v in PARAM_GRID.items()},
    "precision": args.precision,
    "effective_batch_size": BATCH_SIZE,
    "eval_languages": eval_languages,
    "best_config": best_params,
    "evaluations": {},
}

def metrics_for(true_r, pred_r, true_n, pred_n):
    pearson, pearson_p = scipy.stats.pearsonr(true_r, pred_r)
    kendall, kendall_p = scipy.stats.kendalltau(true_r, pred_r)
    spearman, spearman_p = scipy.stats.spearmanr(true_r, pred_r)
    return {
        "n": int(len(true_r)),
        "pearson": float(pearson),
        "pearson_p": float(pearson_p),
        "kendall": float(kendall),
        "kendall_p": float(kendall_p),
        "spearman": float(spearman),
        "spearman_p": float(spearman_p),
        "mse": float(mean_squared_error(true_n, pred_n)),
        "rmse": float(np.sqrt(mean_squared_error(true_n, pred_n))),
        "mae": float(np.mean(np.abs(true_n - pred_n))),
        "r2": float(r2_score(true_n, pred_n)),
        "bias_raw": float(np.mean(pred_r - true_r)),
    }

for language in eval_languages:
    column = LANGUAGE_COLUMNS[language]
    setting = 'in-language' if language == args.language else "transfer zero-shot"
    _, _, predictions = evaluate_model(model, test_loader_for(column))
    predictions_raw = denormalize(predictions)

    entry = {"setting": setting, "column": column, "total": metrics_for(
        true_raw, predictions_raw, true_norm, predictions
    )}
    for source in test_data["source"].unique():
        mask = (test_data["source"] == source).values
        if mask.sum() > 2:
            entry[source] = metrics_for(
                true_raw[mask], predictions_raw[mask], true_norm[mask], predictions[mask]
            )
    results["evaluations"][language] = entry

    total = entry["total"]
    print(
        f"\n=== test on '{column}' ({language}, {setting}) ===\n"
        f"  Pearson={total['pearson']:.3f} (p={total['pearson_p']:.2e})  "
        f"Kendall={total['kendall']:.3f} (p={total['kendall_p']:.2e})\n"
        f"  MSE={total['mse']:.5f}  R2={total['r2']:.4f}"
    )
    for source in test_data["source"].unique():
        if source in entry:
            s = entry[source]
            print(
                f"    {source:<10} n={s['n']:<5} Pearson={s['pearson']:.3f} "
                f"(p={s['pearson_p']:.2e})  Kendall={s['kendall']:.3f}"
            )

    pd.DataFrame(
        {
            "caption_id": test_data["caption_id"].values,
            "source": test_data["source"].values,
            "true_score": true_raw,
            "predicted_score": predictions_raw,
        }
    ).to_csv(os.path.join(PREDICTIONS_DIR, f"{RUN}__on_{language}.csv"), index=False)

with open(os.path.join(RESULTS_DIR, f"{RUN}.json"), "w") as handle:
    json.dump(results, handle, indent=2, ensure_ascii=False)

print(f"\nresults -> {os.path.relpath(os.path.join(RESULTS_DIR, RUN + '.json'), languages.REPO)}")
