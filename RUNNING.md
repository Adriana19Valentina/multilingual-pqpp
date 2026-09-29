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

## The two configurations

Every predictor is run in two configurations. The comparison between them is the
result of the study.

| # | trained on | tested on | what it measures |
|---|---|---|---|
| 1 | English | your language | **zero-shot transfer** — how much of an English-trained predictor survives translation |
| 2 | your language | your language | **in-language** — how much fine-tuning on the translation recovers |

Configuration 1 is `--language english`, configuration 2 is
`--language italian_reviewed`. Each run evaluates its best checkpoint on **every**
language declared in `src/languages.py`, so configuration 1 needs only one run:
you read off the row for your language.

That same run also produces an English-on-English row. Ignore it — we have
already run it, and it is identical for every partner. It is in
`results/romanian_reviewed/`.

The correlation CNN has only one configuration, because it never reads the
prompt: its numbers are the same in every language by construction.

## What you have to run

24 runs in total. Each run is a full grid search over 9 hyperparameter
configurations, picks the best on validation, and evaluates it on the test
split — so 216 trained models altogether. The run count does not change with
prompt coverage; only the time per run does.

| step | configurations | runs | grid | epochs | approx. time each |
|---|---|---|---|---|---|
| 5. BERT | both | 12 (6 targets × 2) | 9 | 15 | 20 min |
| 6. CLIP, generation | both | 4 (2 targets × 2) | 9 | 100 | 2 min |
| 6. CLIP, retrieval | both | 2 (each gives all 4 cells) | 9 | 25 | 20 min, or 3 h with `--train-text-tower` |
| 7. CNN, generation | n/a | 2 | 9 | 25 | 5 min |
| 7. CNN, retrieval | n/a | 4 | 9 | 25 | 1 h |

Times are for an RTX 3090: roughly 9 hours in total, or 14 with
`--train-text-tower`.

The grid is `learning_rate ∈ {1e-5, 5e-5, 1e-4}` × `weight_decay ∈ {0, 0.01, 0.1}`,
the same one used in the paper. Do not change it — the comparison across
languages depends on it.

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

The translation files live here:

<https://drive.google.com/drive/folders/174k389hDM8cno6PrAw59VlB999w65dKX>

Download the one for your language and merge it:

```bash
python src/add_translations.py path/to/final_reviewed_italian.csv
python src/add_translations.py path/to/final_reviewed_italian.csv --write
```

The first call only reports what it would do, the second writes. Several files or
a glob also work: `python src/add_translations.py "translations/*.csv"`.

You do **not** have to split the file into train / validation / test, or sort it,
or keep any particular row order. The script matches each row to the prompt it
belongs to and works out its split.

**What the file needs.** The translation columns must be named
`caption_<language>`, for example `caption_italian_reviewed`. For identifying the
prompts, either is fine:

- `caption_id` and `source` — preferred, exact
- a `caption` column with the original English prompt — used automatically if
  the two above are absent, compared case- and whitespace-insensitively

A `split` column is optional and is **not** used to define the splits: those come
from this repository, and they are the ones the paper's numbers were computed on.
If the column is present the script checks it against them and reports the
agreement, for example `split column: 10200/10200 rows agree with the published
split`. Values like `train_part1`, `train_part2`, `validation` or `dev` are
recognised and normalised. A mismatch is reported as a warning, with a table of
what moved where, because results computed on a different split are comparable
neither with Table 3 nor with the other languages.

The script also checks the English captions, when your file carries them, and
reports any row whose English text differs from ours even though the id matched —
usually an editing artefact, harmless, but worth knowing about.

**You do not need all 10,200 prompts.** Translate as many as you have; the script
reports the count and carries on. Untranslated prompts are then dropped from
every language alike, so the comparison stays row-for-row. Do not reshuffle the
splits themselves — they stay 6,080 train / 2,040 validation / 2,080 test.

## 3. Configure `src/languages.py`

This file is the only place where the language is declared. It already holds one
block per language — Romanian, French, Italian, Hindi, Danish, Arabic. Uncomment
yours and comment out the rest. Italian is active by default:

```python
# --- Italian ---
TARGET_LANGUAGE = "italian_reviewed"

RAW_TARGET_LANGUAGE = None

COLUMNS = {
    "english": "caption",
    "italian_reviewed": "caption_italian_reviewed",
}

LABELS = {
    "english": "English",
    "italian_reviewed": "Italian (reviewed)",
}
```

If your language is not listed, copy any block and change the four names. The
column names must match the ones you merged in step 2.

`TARGET_LANGUAGE` is the variant every model is trained and reported on, so it
is the reviewed one. It also sets the results folder: everything is written to
`results/italian_reviewed/`. To use a different folder name, set
`RESULTS_SUBDIR` in the same file.

`RAW_TARGET_LANGUAGE` is for the unreviewed machine translation, if your file has
that column as well as the reviewed one. It is evaluated but never trained on, and
it adds one comparison row to the tables, showing whether human review of the
translation made a difference. Leave it `None` when you only have the reviewed
column, as in the Italian file.

Every language you declare in `COLUMNS` must be non-empty for a prompt to be
used, so declare only the ones you report.

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

## 5. Fine-tuned BERT — 12 runs

Configuration 1, trained on English, tested on your language:

```bash
for T in glide sdxl clip_p10 clip_rr blip2_p10 blip2_rr; do
    python src/bert/finetunedbert_multilingual.py --language english --target $T
done
```

Configuration 2, trained and tested on your language:

```bash
for T in glide sdxl clip_p10 clip_rr blip2_p10 blip2_rr; do
    python src/bert/finetunedbert_multilingual.py --language italian_reviewed --target $T
done
```

`bert-base-multilingual-cased`, fine-tuned end to end. This predictor sees only
the text, so it is where translation hurts most — expect the largest drop here.

Each run writes `results/italian_reviewed/<target>__italian_reviewed.json`. The
grid can be interrupted and resumed: finished configurations are skipped.

## 6. Fine-tuned CLIP — 6 runs

Generation, one run per system per configuration:

```bash
python src/clip/finetunedclip_multilingual.py --language english          --target glide
python src/clip/finetunedclip_multilingual.py --language english          --target sdxl
python src/clip/finetunedclip_multilingual.py --language italian_reviewed --target glide
python src/clip/finetunedclip_multilingual.py --language italian_reviewed --target sdxl
```

Retrieval, one run per configuration, each filling all four cells:

```bash
python src/clip/finetunedclip_retrieval.py \
    --language english --encoder xlmr-vitb32 \
    --variant b --features interaction --train-text-tower

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

## 7. Correlation CNN — 6 runs, one configuration

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

The tables carry one row per configuration:

```
English                  <- ignore, already published
  pivot -> target        <- configuration 1, zero-shot transfer
  pivot -> target (raw MT)
Italian (reviewed)       <- configuration 2, in-language
  target -> pivot
```

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
