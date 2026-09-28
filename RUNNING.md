# Running the three predictors

Instructions only. Every command is run from the repository root.

---

## 1. Setup

```bash
git clone https://github.com/Adriana19Valentina/multilingual-pqpp.git
cd multilingual-pqpp
pip install -r requirements.txt
python src/fetch_precomputed.py --bundle xlmr longclip
```

## 2. Merge your translations

Your translation file needs `caption_id` and `source`; the translation columns
must be named `caption_<language>`. For example:

```
caption_id, source, caption, caption_italian, caption_italian_reviewed
```

Merge them into the three split files (one file, or several, or a glob):

```bash
python src/add_translations.py path/to/italian.csv
python src/add_translations.py path/to/italian.csv --write
```

The first call only reports what it would do. The join is on
`(caption_id, source)`, so row order does not matter and nothing is written
unless every row of all three splits is covered.

## 3. Configure `src/languages.py`

Edit these four settings:

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
never trained on, and it adds one comparison row to the tables. Set it to `None`
if you do not have that column.

Then check the configuration:

```bash
python src/languages.py
```

It must print `ok` for every language on all three splits.

## 4. Text embeddings

```bash
python src/clip/compute_text_embeddings.py --encoder xlmr-vitb32
```

## 5. Fine-tuned BERT

```bash
for T in glide sdxl clip_p10 clip_rr blip2_p10 blip2_rr; do
    python src/bert/finetunedbert_multilingual.py --language italian_reviewed --target $T
done
```

## 6. Fine-tuned CLIP

Generation:

```bash
python src/clip/finetunedclip_multilingual.py --language italian_reviewed --target glide
python src/clip/finetunedclip_multilingual.py --language italian_reviewed --target sdxl
```

Retrieval (one run produces all four retrieval cells):

```bash
python src/clip/finetunedclip_retrieval.py \
    --language italian_reviewed --encoder xlmr-vitb32 \
    --variant b --features interaction --train-text-tower
```

Drop `--train-text-tower` for a faster first pass.

## 7. Correlation CNN

Generation:

```bash
python src/correlation_cnn_generative.py --target glide --embed-tag longclip-b --matrix images
python src/correlation_cnn_generative.py --target sdxl  --embed-tag longclip-b --matrix images
```

Retrieval:

```bash
for T in clip_p10 clip_rr blip2_p10 blip2_rr; do
    python src/correlation_cnn_retrieval.py --target $T --encoder longclip-b
done
```

## 8. Tables

```bash
python src/export_results.py
python src/export_clip_table.py
python src/export_corrcnn_table.py
python src/export_retrieval_variants.py
```

Each writes `.txt`, `.tsv` and `.json` into `results/italian_reviewed/`. Add
`--decimal dot` for an English locale.

Rows you have not run appear as `--`.

---

## Notes

- For fine-tuned CLIP always pass `--encoder xlmr-vitb32`. `longclip-b` is
  English-only and will silently mangle other languages.
- The correlation CNN never reads the prompt text, so `--encoder longclip-b` /
  `--embed-tag longclip-b` is correct there.
- Fine-tuned BERT uses `bert-base-multilingual-cased` and has no encoder flag.
- Report retrieval results on the `mscoco` subset.
- Our English and Romanian results are in `results/romanian_reviewed/` for
  comparison.
