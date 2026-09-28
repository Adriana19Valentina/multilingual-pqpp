import argparse
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "dataset")
EMBED_DIR = os.path.join(DATA_DIR, "clip_embeddings")
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GT_DIR = os.path.join(ROOT, "dataset", "retrieval", "ground_truth")

MODEL_ID = "openai/clip-vit-base-patch32"
CORPUS_FILE = "corpus_embeddings_clip-vitb32-hf.npz"
IMAGES_DIR = os.path.join(DATA_DIR, "all")

class ClipRetriever:

    def __init__(self, device=None, corpus_path=None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = CLIPModel.from_pretrained(MODEL_ID).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)

        corpus = np.load(corpus_path or os.path.join(EMBED_DIR, CORPUS_FILE))
        self.image_ids = corpus["image_id"]

        self.corpus = torch.nn.functional.normalize(
            torch.from_numpy(corpus["embeddings"]).to(self.device), dim=-1
        )

    def encode(self, prompts, batch_size=256):
        chunks = []
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                tokens = self.processor(
                    text=prompts[start : start + batch_size], return_tensors="pt",
                    padding=True, truncation=True,
                ).to(self.device)
                chunks.append(
                    self.model.get_text_features(
                        input_ids=tokens["input_ids"],
                        attention_mask=tokens["attention_mask"],
                    ).float()
                )
        return torch.nn.functional.normalize(torch.cat(chunks), dim=-1)

    def retrieve(self, prompts, top_k=25, batch_size=64, with_scores=False):
        queries = self.encode(prompts)
        ids, scores = [], []
        with torch.no_grad():
            for start in range(0, len(queries), batch_size):
                similarity = queries[start : start + batch_size] @ self.corpus.T
                top = similarity.topk(top_k, dim=1)
                ids.append(self.image_ids[top.indices.cpu().numpy()])
                scores.append(top.values.cpu().numpy())
        ids = np.concatenate(ids)
        return (ids, np.concatenate(scores)) if with_scores else ids

    def image_path(self, image_id):
        return os.path.join(IMAGES_DIR, f"{int(image_id):012d}.jpg")

def scores_for(retrieved_ids, relevant, precision_at=10):
    relevant = list(relevant)
    hits = np.isin(retrieved_ids[:precision_at], relevant)
    positions = np.flatnonzero(np.isin(retrieved_ids, relevant))
    return float(hits.mean()), (1.0 / (positions[0] + 1) if len(positions) else 0.0)

def _split_prompts(split):
    import pandas as pd

    frame = pd.read_csv(os.path.join(DATA_DIR, f"pqpp_multilingual_{split}.csv"))
    reference = pd.read_csv(
        os.path.join(GT_DIR, "clip", f"clip_retrieval_{split}_results.csv")
    )
    return frame, reference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", action="append",
                        help='prompt in the pivot language; may be repeated')
    parser.add_argument("--split", choices=["train", "val", "test"],
                        help='run on every query of a split')
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--out", help='save the ids to an .npz file')
    parser.add_argument("--validate", action="store_true",
                        help='compare P@10 and RR against the published targets')
    args = parser.parse_args()

    if not args.prompt and not args.split:
        parser.error("da fie --prompt, fie --split")

    retriever = ClipRetriever()

    if args.prompt:
        ids, scores = retriever.retrieve(args.prompt, args.top_k, with_scores=True)
        for prompt, row, row_scores in zip(args.prompt, ids, scores):
            print(f"\n{prompt}")
            for rank, (image_id, score) in enumerate(zip(row, row_scores), start=1):
                print(f"  {rank:>3}. {int(image_id):012d}  cos={score:.4f}  "
                      f"{retriever.image_path(image_id)}")
        return

    import pickle

    import pandas as pd

    frame, reference = _split_prompts(args.split)
    prompts = [str(x) for x in frame["caption"]]
    print(f"split={args.split}  query-uri={len(prompts)}  top_k={args.top_k}")

    depth = max(args.top_k, 2000) if args.validate else args.top_k
    ids = retriever.retrieve(prompts, depth)

    if args.out:
        np.savez_compressed(args.out, **{f"{args.split}_top{args.top_k}":
                                         ids[:, : args.top_k]})
        print(f"wrote: {args.out}")

    if args.validate:
        relevance = {
            (item["source"], int(item["index"])): {int(x) for x in item["gt"]}
            for item in pickle.load(
                open(os.path.join(GT_DIR, f"retrieval_{args.split}_gt.pickle"), "rb")
            )
        }
        computed = [
            scores_for(ids[position],
                       relevance.get((frame.iloc[position]["source"],
                                      int(reference.iloc[position]["index"])), set()))
            for position in range(len(frame))
        ]
        computed = pd.DataFrame(computed, columns=["precision", "reciprocal_rank"])
        print(f"\n{'metric':<18}{'identic':>9}{'r':>9}   medii")
        for metric in ["precision", "reciprocal_rank"]:
            a = computed[metric].to_numpy()
            b = reference[metric].to_numpy()
            print(f"{metric:<18}{np.isclose(a, b, atol=1e-9).mean():>8.1%}"
                  f"{np.corrcoef(a, b)[0, 1]:>9.4f}   {a.mean():.4f} vs {b.mean():.4f}")

if __name__ == "__main__":
    main()
