# Running the three predictors

## What this is for

PQPP (Poesina et al., CVPR 2025) predicts how well a text-to-image system will
answer a prompt, *before* it runs. It does this on English prompts only. This
repository extends the benchmark to other languages, to measure how much of the
prediction quality survives translation.

Each partner takes the same 10,200 prompts, translated into one language, and
runs the same three predictors. Everything else — the splits, the images, the
retrieval lists, the ground truth — is identical and ships with this
repository.

You do not need to translate all 10,200 prompts, and you do not need to split
them into train / validation / test — see *Prompt coverage* below.

**What you produce:** 18 cells, i.e. the multilingual version of Table 3 of the
paper. Three predictors × six targets:

| predictor | when it runs | inputs it sees |
|---|---|---|
| Fine-tuned BERT | before generation / retrieval | prompt text only |
| Fine-tuned CLIP | after generation / retrieval | prompt text + images |
| Correlation CNN | after generation / retrieval | images only |

The six targets are two generative systems (GLIDE, SDXL, scored by human
judgement) and two retrieval systems × two metrics (CLIP and BLIP-2, P@10 and
RR).

## What you have to run

15 runs in total. Each run is a full grid search over 9 hyperparameter
configurations, picks the best on validation, and evaluates it on the test
split — so 135 trained models altogether. The run count does not change with
prompt coverage; only the time per run does.

| step | runs | grid | epochs | approx. time each |
|---|---|---|---|---|
| 5. BERT | 6 (one per target) | 9 | 15 | 20 min |
| 6. CLIP, generation | 2 | 9 | 100 | 2 min |
| 6. CLIP, retrieval | 1 (gives all 4 cells) | 9 | 25 | 20 min, or 3 h with `--train-text-tower` |
| 7. CNN, generation | 2 | 9 | 25 | 5 min |
| 7. CNN, retrieval | 4 | 9 | 25 | 1 h |

Times are for an RTX 3090; roughly 8 hours for everything.

The grid is `learning_rate ∈ {1e-5, 5e-5, 1e-4}` × `weight_decay ∈ {0, 0.01, 0.1}`,
the same one used in the paper. Do not change it — the comparison across
languages depends on it.

Every run evaluates its best checkpoint on **all** languages declared in
`src/languages.py`, not just the one it trained on. So a single run gives you
both the in-language row and the transfer row back to English.

You do not need to train on English. Our English results are already in
`results/romanian_reviewed/` and are the same for every partner.

## What you do not have to compute

These ship precomputed or in the repository, because they do not depend on the
language:

- image embeddings for the 40,800 generated images and the 94,942 retrieved ones
- the CLIP and BLIP-2 retrieval lists, reconstructed and validated against the
  published scores (99.3% and 96.6% exact match)
- the relevance labels used to train the retrieval predictor
- the per-query P@10 and RR ground truth

Only the text embeddings depend on your language, and they take seconds
(step 4).

## Prompt coverage

There is no requirement to translate every prompt, and no extra step if you do
not. A prompt is used only if it has a non-empty translation in **every**
language declared in `COLUMNS`; the rest are dropped from all splits, for all
languages, in all three predictors, automatically. Every script prints the count
before it starts:

```
coverage:
  train: 4261/6080 prompts translated in every language, 1819 dropped
  val: 1720/2040 prompts translated in every language, 320 dropped
  test: 1627/2080 prompts translated in every language, 453 dropped
```

Dropping the same rows everywhere is what keeps the study valid. Within one
language, the in-language row, the transfer row and the raw-MT row are then all
computed on identical prompts, so the differences between them mean something.

Across languages they do not: if Italian evaluates on 1,627 test prompts and
Romanian on 2,080, the two absolute Pearson values are not directly comparable.
What is comparable is each language's own *retention* — its score divided by the
English score from the same run, over the same prompts. Report that, and report
the prompt count alongside every table. `export_results.py` writes it into the
header and into `test_prompts_evaluated` in the JSON.

Because every declared language shrinks the usable set, declare only the ones
you actually report. If you have no reviewed/unreviewed distinction, set
`RAW_TARGET_LANGUAGE = None` and leave that column out of `COLUMNS`.

---

# Steps

Every command is run from the repository root.

## 1. Setup

```bash
git clone https://github.com/Adriana19Valentina/multilingual-pqpp.git
cd multilingual-pqpp
pip install -r requirements.txt
python src/fetch_precomputed.py --bundle xlmr longclip
```

The download is about 420 MB. Check what you already have with
`python src/fetch_precomputed.py --check`.

## 2. Merge your translations

You do **not** have to split your translations into train / validation / test.
One file holding all 10,200 prompts is the normal case; the script works out
which split each prompt belongs to.

```bash
python src/add_translations.py path/to/italian.csv
python src/add_translations.py path/to/italian.csv --write
```

The first call only reports what it would do, the second writes. Several files or
a glob also work: `python src/add_translations.py "translations/*.csv"`.

**What the file needs.** The translation columns must be named
`caption_<language>`, for example `caption_italian` and
`caption_italian_reviewed`. For identifying the prompts, either is fine:

- `caption_id` and `source` — preferred, exact
- a `caption` column with the original English prompt — used automatically if
  the two above are absent, compared case- and whitespace-insensitively

Row order never matters. A prompt whose English text was edited will not match
on a caption join; the script reports how many such rows it ignored.

**You do not need all 10,200 prompts.** Translate as many as you have; the
script reports the count and carries on. The splits themselves stay fixed —
6,080 train / 2,040 validation / 2,080 test — so do not reshuffle them, but a
partly filled column is fine.

## 3. Configure `src/languages.py`

This file is the only place where the language is declared. Edit these four
settings:

```python
TARGET_LANGUAGE = "italian_reviewed"

RAW_TARGET_LANGUAGE = "italian"

COLUMNS = {
    "english": "caption",
    "italian": "caption_italian",
    "italian_reviewed": "caption_italian_reviewed",
}

LABELS = {
    "english": "English",
    "italian": "Italian (raw MT)",
    "italian_reviewed": "Italian (reviewed)",
}
```

`TARGET_LANGUAGE` is the variant every model is trained and reported on, so it
is the reviewed one. It also sets the results folder: everything is written to
`results/italian_reviewed/`. To use a different folder name, set
`RESULTS_SUBDIR` in the same file.

`RAW_TARGET_LANGUAGE` is the unreviewed machine translation. It is evaluated but
never trained on, and it adds one comparison row to the tables — that row is
what shows whether human review of the translation matters. Set it to `None` if
you do not have that column.

Then check the configuration:

```bash
python src/languages.py
```

It must print `ok` for every language on all three splits.

## 4. Text embeddings

```bash
python src/clip/compute_text_embeddings.py --encoder xlmr-vitb32
```

One pass over 10,200 prompts per language. Takes seconds. Needed by steps 6
and 7.

## 5. Fine-tuned BERT — 6 runs

```bash
for T in glide sdxl clip_p10 clip_rr blip2_p10 blip2_rr; do
    python src/bert/finetunedbert_multilingual.py --language italian_reviewed --target $T
done
```

`bert-base-multilingual-cased`, fine-tuned end to end. This predictor sees only
the text, so it is where translation hurts most — expect the largest drop here.

Each run writes `results/italian_reviewed/<target>__italian_reviewed.json`. The
grid can be interrupted and resumed: finished configurations are skipped.

## 6. Fine-tuned CLIP — 3 runs

Generation, one run per system:

```bash
python src/clip/finetunedclip_multilingual.py --language italian_reviewed --target glide
python src/clip/finetunedclip_multilingual.py --language italian_reviewed --target sdxl
```

Retrieval, one run for all four cells:

```bash
python src/clip/finetunedclip_retrieval.py \
    --language italian_reviewed --encoder xlmr-vitb32 \
    --variant b --features interaction --train-text-tower
```

The retrieval model classifies (query, image) pairs over the 50 retrieved images
per query, then aggregates into P@10 and RR — which is why one run fills all
four cells.

The three flags are worth about +0.14 to +0.20 Pearson over the original
formulation. Drop `--train-text-tower` for a much faster first pass; the other
two cost nothing.

## 7. Correlation CNN — 6 runs

Generation:

```bash
python src/correlation_cnn_generative.py --target glide --embed-tag longclip-b --matrix images
python src/correlation_cnn_generative.py --target sdxl  --embed-tag longclip-b --matrix images
```

Retrieval, one run per target:

```bash
for T in clip_p10 clip_rr blip2_p10 blip2_rr; do
    python src/correlation_cnn_retrieval.py --target $T --encoder longclip-b
done
```

This predictor never reads the prompt, only a correlation matrix over the
images, so its numbers are identical in every language. Run it anyway: it is the
control that shows the drop in the other two predictors comes from the text and
not from something else.

## 8. Tables

```bash
python src/export_results.py
python src/export_clip_table.py
python src/export_corrcnn_table.py
python src/export_retrieval_variants.py
```

Each writes `.txt`, `.tsv` and `.json` into `results/italian_reviewed/`. Add
`--decimal dot` for an English locale.

Cells you have not run appear as `--`. For BERT, the missing commands are listed
explicitly at the end of the table.

---

## Notes

- For fine-tuned CLIP always pass `--encoder xlmr-vitb32`. `longclip-b` is
  English-only: its vocabulary has no diacritics, so it shreds a translated
  prompt into byte fragments without raising any error.
- The correlation CNN never reads the prompt text, so `--encoder longclip-b` /
  `--embed-tag longclip-b` is correct there, and is the better encoder.
- Fine-tuned BERT uses `bert-base-multilingual-cased` and has no encoder flag.
- Report retrieval results on the `mscoco` subset. The 200 DrawBench prompts do
  not describe existing photographs, and the paper's per-prompt corpus for them
  was never published.
- Our English and Romanian results are in `results/romanian_reviewed/`.
