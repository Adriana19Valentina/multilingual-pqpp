import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import languages

DATA_DIR = languages.DATA_DIR
EMBED_DIR = languages.EMBED_DIR
LANGUAGE_COLUMNS = languages.COLUMNS
SPLIT_FILES = languages.SPLIT_FILES
FT_ENCODER = f"xlmr-vitb32-ft-{languages.TARGET_LANGUAGE}"
TOP_K = 25
PARAM_GRID = {"learning_rate": [1e-5, 1e-4, 5e-5], "weight_decay": [0, 0.1, 0.01]}
NUM_EPOCHS = 25
BATCH_SIZE = 256
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--language", required=True, choices=sorted(LANGUAGE_COLUMNS),
                    help='primary language; appears first in the training set')
parser.add_argument(
    "--augment", nargs="*", default=[], choices=sorted(LANGUAGE_COLUMNS),
    help='languages ADDED to training on top of --language. Every prompt contributes pairs in each language, so the training set grows proportionally. Relevance labels were defined in the pivot language, so a translation cannot REPLACE it, but it can complement it as a paraphrase.')
parser.add_argument(
    "--init-from", default=None,
    help='start from a checkpoint saved by an earlier run instead of random initialization. This gives SEQUENTIAL fine-tuning: the model first learns the task in the pivot language, the one the labels were defined in, and only then adapts to the surface form of the translation. The learning rate is scaled down automatically via --init-lr-scale.')
parser.add_argument("--init-lr-scale", type=float, default=0.2,
                    help='factor applied to the learning rate after initializing from a checkpoint; smaller, so that what was learned is not forgotten')
parser.add_argument(
    "--consistency", type=float, default=0.0,
    help='weight of the term forcing the same prediction across languages, computed as MSE between the outputs for one prompt in two languages; 0 disables it')
parser.add_argument("--encoder", required=True,
                    choices=["longclip-b", "xlmr-vitb32", "xlmr-vith14",
                             FT_ENCODER])
parser.add_argument(
    "--features", default="concat", choices=["concat", "interaction"],
    help="'concat' is [text ; image], as in the original. 'interaction' adds the element-wise product and the absolute difference: [t ; i ; t*i ; |t-i|]. This makes the text-image ALIGNMENT explicit instead of leaving the network to discover it, so the quality of the text representation matters far more.")
parser.add_argument(
    "--train-text-tower", action="store_true",
    help='train the text tower together with the head, at a much smaller learning rate. Without it, fine-tuning on a translation adapts nothing language-sensitive: the encoder stays frozen and the head merely learns over fixed embeddings.')
parser.add_argument(
    "--variant", default="a", choices=["a", "b", "c"],
    help='a is faithful to the original code, with the median threshold and the rank offset on BLIP-2 RR; b aggregates through expected values, with no threshold and no offset; c is b plus rank as an input feature',
)
args = parser.parse_args()

TRAIN_LANGUAGES = list(dict.fromkeys([args.language] + args.augment))
RUN = (f"retrieval__{'+'.join(TRAIN_LANGUAGES)}"
       + ("" if args.variant == "a" else f"__{args.variant}")
       + ("__inter" if args.features == "interaction" else "")
       + ("__tt" if args.train_text_tower else "")
       + (f"__cons{args.consistency:g}" if args.consistency else "")
       + ("__seq" if args.init_from else ""))

USE_RANK = args.variant == "c"

suffix = "clip_retrieval" if args.encoder == "xlmr-vitb32" else f"clip_retrieval_{args.encoder}"
RESULTS_DIR = os.path.join(languages.RESULTS_DIR, suffix)
PREDICTIONS_DIR = os.path.join(languages.PREDICTIONS_DIR, suffix)
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
image_features = image_npz["embeddings"]
EMBED_DIM = image_features.shape[1]

FEATURE_BLOCKS = 4 if args.features == "interaction" else 2
INPUT_DIM = FEATURE_BLOCKS * EMBED_DIM + (2 if USE_RANK else 0)

lists = {
    "clip": np.load(os.path.join(EMBED_DIR, "retrieval_lists_clip.npz")),
    "blip2": np.load(os.path.join(EMBED_DIR, "retrieval_lists_blip2.npz")),
}

relevance_labels = np.load(os.path.join(EMBED_DIR, "retrieval_labels.npz"))

frames = {split: pd.read_csv(languages.split_path(split)) for split in SPLIT_FILES}

masks = {}
print("coverage:")
for split in SPLIT_FILES:
    masks[split] = languages.usable_mask(frames[split])
    languages.report_coverage(split, masks[split])

lists = {
    system: {f"{split}_top{TOP_K}": data[f"{split}_top{TOP_K}"][masks[split]]
             for split in SPLIT_FILES}
    for system, data in lists.items()
}
relevance_labels = {
    f"{split}_labels": relevance_labels[f"{split}_labels"][masks[split]]
    for split in SPLIT_FILES
}
frames = {split: frame[masks[split]].reset_index(drop=True)
          for split, frame in frames.items()}

if args.encoder == "longclip-b":
    import sys
    sys.path.insert(0, languages.THIRD_PARTY)
    from huggingface_hub import hf_hub_download
    from longclip_model import longclip

    text_model, _ = longclip.load(
        hf_hub_download("BeichenZhang/LongCLIP-B", "longclip-B.pt"), device=device
    )
    text_model.eval()
    assert args.language == languages.PIVOT, \
        'Long-CLIP is monolingual; use it only on the pivot language'

    def encode_text(texts):
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), 256):
                tokens = longclip.tokenize(texts[start : start + 256]).to(device)
                out.append(text_model.encode_text(tokens).float().cpu().numpy())
        return np.concatenate(out)
else:
    import open_clip

    _name, _pretrained = {
        "xlmr-vitb32": ("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),
        "xlmr-vith14": ("xlm-roberta-large-ViT-H-14", "frozen_laion5b_s13b_b90k"),
        FT_ENCODER: ("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),
    }[args.encoder]
    text_model, _, _ = open_clip.create_model_and_transforms(_name, pretrained=_pretrained)
    if "-ft-" in args.encoder:

        _lang = args.encoder.split("-ft-")[1]
        _ck = torch.load(os.path.join(languages.CHECKPOINT_ROOT, "clip",
                                     f"clip_finetuned_{_lang}.pt"),
                         map_location="cpu", weights_only=False)
        text_model.load_state_dict(_ck["model_state_dict"])
        print(f"adapted text weights: R@1 "
              f"{_ck['baseline_recall@1']:.3f} -> {_ck['recall@1']:.3f}")
    tokenizer = open_clip.get_tokenizer(_name)
    text_model.eval().to(device)

    def encode_text(texts):
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), 256):
                tokens = tokenizer(texts[start : start + 256]).to(device)
                out.append(text_model.encode_text(tokens).float().cpu().numpy())
        return np.concatenate(out)

def compose(text_block, image_block):
    blocks = [text_block, image_block]
    if args.features == "interaction":
        blocks += [text_block * image_block, (text_block - image_block).abs()
                   if torch.is_tensor(text_block) else np.abs(text_block - image_block)]
    return (torch.cat(blocks, dim=-1) if torch.is_tensor(text_block)
            else np.hstack(blocks))

def image_matrix(split):
    frame = frames[split]
    rows = np.stack([
        [row_of_image[int(i)] for i in np.concatenate([
            lists["clip"][f"{split}_top{TOP_K}"][position],
            lists["blip2"][f"{split}_top{TOP_K}"][position]])]
        for position in range(len(frame))
    ])
    return torch.from_numpy(image_features[rows.reshape(-1)]).view(len(frame), 50, -1)

def labels_for(split):
    return torch.tensor(relevance_labels[f"{split}_labels"], dtype=torch.float32)

def tokens_for(split, language=None):
    column = LANGUAGE_COLUMNS[language or args.language]
    return tokenizer([str(x) for x in frames[split][column]])

def build(split, language=None):
    frame = frames[split]
    texts = encode_text([str(x) for x in frame[LANGUAGE_COLUMNS[language or args.language]]])
    split_labels = relevance_labels[f"{split}_labels"]

    features, labels = [], []
    for position in range(len(frame)):
        retrieved = np.concatenate([
            lists["clip"][f"{split}_top{TOP_K}"][position],
            lists["blip2"][f"{split}_top{TOP_K}"][position],
        ])
        rows = [row_of_image[int(i)] for i in retrieved]
        text_block = np.tile(texts[position], (50, 1))
        image_block = image_features[rows]
        block = [compose(text_block, image_block)]
        if USE_RANK:

            ranks = np.tile(np.arange(1, TOP_K + 1, dtype=np.float32), 2)
            block += [(ranks / TOP_K)[:, None], (1.0 / ranks)[:, None]]
        features.append(np.hstack(block))
        labels.append(split_labels[position].astype(np.float32))
    return (
        torch.from_numpy(np.concatenate(features).astype(np.float32)).to(device),
        torch.from_numpy(np.concatenate(labels).astype(np.float32)).to(device),
    )

if args.train_text_tower:

    train_x = train_y = val_x = val_y = None
    print('TRAINABLE text tower: text is re-encoded at every step')
else:
    parts = [build("train", language) for language in TRAIN_LANGUAGES]
    train_x = torch.cat([x for x, _ in parts])
    train_y = torch.cat([y for _, y in parts])
    del parts

    val_x, val_y = build("val")
    if len(TRAIN_LANGUAGES) > 1:
        print(f"  training on {len(TRAIN_LANGUAGES)} languages: {', '.join(TRAIN_LANGUAGES)}")
print(f"run={RUN}  encoder={args.encoder}  language={args.language}")
if train_x is not None:
    print(f"  train={tuple(train_x.shape)}  positives={train_y.mean():.1%}")

class NeuralNetworkClassifier(nn.Module):
    def __init__(self):
        super(NeuralNetworkClassifier, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x.view(-1, INPUT_DIM))))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.sigmoid(self.fc3(x))

def predict(model, features):
    model.eval()
    with torch.no_grad():
        return torch.cat(
            [model(features[i : i + 8192]).squeeze(1) for i in range(0, len(features), 8192)]
        )

INIT = None
if args.init_from:
    INIT = torch.load(args.init_from, map_location="cpu", weights_only=False)
    assert INIT["input_dim"] == INPUT_DIM, (
        f"checkpointul are input_dim={INIT['input_dim']}, rularea curenta {INPUT_DIM}")
    print(f"initialised from {os.path.basename(args.init_from)} "
          f"(antrenat pe {'+'.join(INIT['train_languages'])}), "
          f"lr x{args.init_lr_scale:g}")

def train_one_config(lr, decay):
    set_seed(SEED)
    model = NeuralNetworkClassifier().to(device)
    loss_fn = nn.BCELoss()
    if INIT is not None:
        model.load_state_dict(INIT["model_state_dict"])
        if "text_state_dict" in INIT:
            text_model.load_state_dict(INIT["text_state_dict"])
        lr = lr * args.init_lr_scale

    if not args.train_text_tower:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        best = {"val_loss": float("inf")}

        block = len(train_x) // len(TRAIN_LANGUAGES)
        for epoch in range(NUM_EPOCHS):
            model.train()
            order = torch.randperm(len(train_x), device=device)
            for start in range(0, len(order), BATCH_SIZE):
                batch = order[start : start + BATCH_SIZE]
                predictions = model(train_x[batch]).squeeze(1)
                loss = loss_fn(predictions, train_y[batch])
                if args.consistency and len(TRAIN_LANGUAGES) > 1:

                    other = (batch + block) % len(train_x)
                    with_other = model(train_x[other]).squeeze(1)
                    loss = loss + args.consistency * nn.functional.mse_loss(
                        predictions, with_other)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_loss = float(loss_fn(predict(model, val_x), val_y))
            if val_loss < best["val_loss"]:
                best = {"learning_rate": lr, "weight_decay": decay, "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "state_dict": {k: v.clone() for k, v in model.state_dict().items()}}
        del model
        torch.cuda.empty_cache()
        return best

    for parameter in text_model.parameters():
        parameter.requires_grad_(True)
    text_model.train()
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": lr},

        {"params": text_model.parameters(), "lr": lr / 50},
    ], weight_decay=decay)

    train_tokens = tokens_for("train").to(device)
    train_images = image_matrix("train").to(device)
    train_labels = labels_for("train").to(device)
    val_tokens = tokens_for("val").to(device)
    val_images = image_matrix("val").to(device)
    val_labels = labels_for("val").to(device)
    QUERY_BATCH = 16

    def forward(tokens, images):
        text = text_model.encode_text(tokens).float()
        text = text.unsqueeze(1).expand(-1, 50, -1)
        return model(compose(text, images).reshape(-1, INPUT_DIM)).view(len(tokens), 50)

    best = {"val_loss": float("inf")}
    for epoch in range(NUM_EPOCHS):
        model.train()
        text_model.train()
        order = torch.randperm(len(train_tokens))
        for start in range(0, len(order), QUERY_BATCH):
            batch = order[start : start + QUERY_BATCH]
            predictions = forward(train_tokens[batch], train_images[batch])
            loss = loss_fn(predictions, train_labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        text_model.eval()
        with torch.no_grad():
            losses = [
                float(loss_fn(forward(val_tokens[i : i + QUERY_BATCH],
                                      val_images[i : i + QUERY_BATCH]),
                              val_labels[i : i + QUERY_BATCH]))
                for i in range(0, len(val_tokens), QUERY_BATCH)
            ]
        val_loss = float(np.mean(losses))
        print(f"    epoch {epoch + 1}/{NUM_EPOCHS} val_BCE={val_loss:.5f}", flush=True)
        if val_loss < best["val_loss"]:
            best = {"learning_rate": lr, "weight_decay": decay, "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                    "text_state_dict": {k: v.detach().cpu().clone()
                                        for k, v in text_model.state_dict().items()}}
    torch.cuda.empty_cache()
    return best

best_overall = {"val_loss": float("inf")}
grid = []
for lr in PARAM_GRID["learning_rate"]:
    for decay in PARAM_GRID["weight_decay"]:
        config = train_one_config(lr, decay)
        grid.append({k: v for k, v in config.items() if k != "state_dict"})
        print(f"  lr={lr:g} wd={decay:g}  epoch {config['epoch']:>2}  "
              f"val_BCE={config['val_loss']:.5f}", flush=True)
        if config["val_loss"] < best_overall["val_loss"]:
            best_overall = config

best_config = {k: v for k, v in best_overall.items()
               if k not in ("state_dict", "text_state_dict")}
print(f"\nBest: lr={best_config['learning_rate']:g} wd={best_config['weight_decay']:g} "
      f"epoch {best_config['epoch']}  val_BCE={best_config['val_loss']:.5f}")

model = NeuralNetworkClassifier().to(device)
model.load_state_dict(best_overall["state_dict"])
if "text_state_dict" in best_overall:
    text_model.load_state_dict(best_overall["text_state_dict"])
    text_model.eval()

def aggregate_expected(prediction_list):
    out = {"clip_p10": [], "clip_rr": [], "blip2_p10": [], "blip2_rr": []}
    for i in range(0, len(prediction_list), 50):
        for system, offset in [("clip", i), ("blip2", i + 25)]:
            probabilities = np.asarray(prediction_list[offset : offset + 25], dtype=float)
            out[f"{system}_p10"].append(probabilities[:10].mean())
            survival = np.concatenate([[1.0], np.cumprod(1.0 - probabilities)[:-1]])
            ranks = np.arange(1, len(probabilities) + 1)
            out[f"{system}_rr"].append(float((probabilities * survival / ranks).sum()))
    return {k: np.array(v) for k, v in out.items()}

def aggregate(prediction_list):
    limit = np.quantile(prediction_list, 0.5)
    out = {"clip_p10": [], "clip_rr": [], "blip2_p10": [], "blip2_rr": []}
    for i in range(0, len(prediction_list), 50):
        clip_rr = 0.0
        for j in range(i, i + 25):
            if prediction_list[j] >= limit:
                clip_rr = 1 / (j - i + 1)
                break
        clip_p10 = sum(prediction_list[j] >= limit for j in range(i, i + 10)) / 10

        blip2_rr = 0.0
        for j in range(i + 25, i + 50):
            if prediction_list[j] >= limit:
                blip2_rr = 1 / (j - i + 1)
                break
        blip2_p10 = sum(prediction_list[j] >= limit for j in range(i + 25, i + 35)) / 10

        out["clip_p10"].append(clip_p10)
        out["clip_rr"].append(clip_rr)
        out["blip2_p10"].append(blip2_p10)
        out["blip2_rr"].append(blip2_rr)
    return {k: np.array(v) for k, v in out.items()}

test_frame = frames["test"]

eval_languages = ([languages.PIVOT] if args.encoder == "longclip-b"
                  else list(LANGUAGE_COLUMNS))
targets = {
    "clip_p10": test_frame["p10_clip"].to_numpy(),
    "clip_rr": test_frame["rr_clip"].to_numpy(),
    "blip2_p10": test_frame["p10_blip2"].to_numpy(),
    "blip2_rr": test_frame["rr_blip2"].to_numpy(),
}

def evaluate_target(target, values):
    entry = {}
    for subset, mask in [("total", np.ones(len(frames["test"]), bool)),
                         ("mscoco", (frames["test"]["source"] == "mscoco").to_numpy()),
                         ("drawbench", (frames["test"]["source"] == "drawbench").to_numpy())]:
        if mask.sum() < 3:
            continue
        a, b = values[mask], targets[target][mask]
        if a.std() == 0 or b.std() == 0:
            entry[subset] = {"n": int(mask.sum()), "pearson": 0.0, "pearson_p": 1.0,
                             "kendall": 0.0, "kendall_p": 1.0, "degenerate": True}
            continue
        pearson, pearson_p = scipy.stats.pearsonr(b, a)
        kendall, kendall_p = scipy.stats.kendalltau(b, a)
        entry[subset] = {
            "n": int(mask.sum()), "pearson": float(pearson), "pearson_p": float(pearson_p),
            "kendall": float(kendall), "kendall_p": float(kendall_p),
            "mean_predicted": float(a.mean()), "mean_target": float(b.mean()),
        }
    return entry

results = {
    "run": RUN, "predictor": "finetuned_clip_retrieval", "encoder": args.encoder,
    "language": args.language, "grid_complete": len(grid) == 9,
    "param_grid": PARAM_GRID, "num_epochs": NUM_EPOCHS, "best_config": best_config,
    "variant": args.variant, "features": args.features,
    "train_languages": TRAIN_LANGUAGES, "consistency": args.consistency,
    "init_from": args.init_from, "init_lr_scale": args.init_lr_scale if args.init_from else None,
    "train_text_tower": args.train_text_tower,
    "uses_rank_feature": USE_RANK,
    "aggregation_note": (
        'reproduces compute_predictions.py, including the rank offset on BLIP-2 RR and the median threshold computed on the test set' if args.variant == "a"
        else 'expected values from probabilities; no threshold, no rank offset'),
    "eval_languages": eval_languages,
    "evaluations": {},
}

for language in eval_languages:
    test_x, _ = build("test", language)
    raw = predict(model, test_x).cpu().numpy()
    aggregated = aggregate(raw) if args.variant == "a" else aggregate_expected(raw)
    del test_x
    torch.cuda.empty_cache()
    setting = 'in-language' if language == args.language else "transfer zero-shot"
    print(f"\n=== test pe {language} ({setting}) ===")
    per_language = {"setting": setting}
    for target, values in aggregated.items():
        entry = evaluate_target(target, values)
        per_language[target] = entry
        total = entry["total"]
        print(f"  {target:<11} Pearson={total['pearson']:+.3f} "
              f"(p={total['pearson_p']:.1e})  Kendall={total['kendall']:+.3f}")
    results["evaluations"][language] = per_language
    if language == args.language:
        pd.DataFrame({
            "caption_id": test_frame["caption_id"].to_numpy(),
            "source": test_frame["source"].to_numpy(),
            **{f"pred_{k}": v for k, v in aggregated.items()},
            **{f"true_{k}": v for k, v in targets.items()},
        }).to_csv(os.path.join(PREDICTIONS_DIR, f"{RUN}.csv"), index=False)

checkpoint = {"model_state_dict": best_overall["state_dict"],
              "config": best_config, "run": RUN, "encoder": args.encoder,
              "features": args.features, "input_dim": INPUT_DIM,
              "train_languages": TRAIN_LANGUAGES}
if "text_state_dict" in best_overall:
    checkpoint["text_state_dict"] = best_overall["text_state_dict"]
torch.save(checkpoint, os.path.join(RESULTS_DIR, f"{RUN}.pth"))

with open(os.path.join(RESULTS_DIR, f"{RUN}.json"), "w") as handle:
    json.dump(results, handle, indent=2, ensure_ascii=False)
print(f"\nresults -> {os.path.relpath(os.path.join(RESULTS_DIR, RUN + '.json'), languages.REPO)}")
