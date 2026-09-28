import argparse
import glob
import os
import sys as _sys

import pandas as pd

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import languages

KEY = ["caption_id", "source"]

parser = argparse.ArgumentParser(
    description="Merge translation columns into data/pqpp_multilingual_{train,val,test}.csv, "
                "joining on (caption_id, source) so row order does not matter.")
parser.add_argument("input", nargs="+",
                    help="CSV files or glob patterns holding the translations; "
                         "each must have caption_id and source")
parser.add_argument("--columns", nargs="*", default=None,
                    help="translation columns to take; default is every column "
                         "starting with 'caption_' that is not already present")
parser.add_argument("--write", action="store_true",
                    help="overwrite the split files; without this the merge is "
                         "only reported")
args = parser.parse_args()

paths = sorted({p for pattern in args.input for p in glob.glob(pattern)})
if not paths:
    raise SystemExit(f"no file matched: {args.input}")

incoming = []
for path in paths:
    frame = pd.read_csv(path)
    missing = [k for k in KEY if k not in frame.columns]
    if missing:
        raise SystemExit(f"{path}: missing key column(s) {missing}")
    incoming.append(frame)
    print(f"read {path}  {len(frame)} rows  columns: {list(frame.columns)}")

merged = pd.concat(incoming, ignore_index=True)
duplicates = merged.duplicated(KEY).sum()
if duplicates:
    raise SystemExit(f"{duplicates} duplicate (caption_id, source) pairs across the inputs")

splits = {s: pd.read_csv(languages.split_path(s)) for s in languages.SPLIT_FILES}
known = set().union(*[set(f.columns) for f in splits.values()])

if args.columns:
    columns = args.columns
else:
    columns = [c for c in merged.columns
               if c.startswith("caption_") and c not in KEY and c not in known]
if not columns:
    raise SystemExit("no new translation column found; pass --columns explicitly")
absent = [c for c in columns if c not in merged.columns]
if absent:
    raise SystemExit(f"requested column(s) absent from the inputs: {absent}")
print(f"\ncolumns to merge: {columns}")

updated = {}
for split, frame in splits.items():
    take = merged[KEY + columns].drop_duplicates(KEY)
    out = frame.merge(take, on=KEY, how="left", suffixes=("", "__incoming"))
    assert len(out) == len(frame), f"{split}: the merge changed the row count"

    report = []
    for column in columns:
        empty = int(out[column].isna().sum())
        report.append(f"{column}={len(out) - empty}/{len(out)}")
        if empty:
            example = out.loc[out[column].isna(), KEY].iloc[0].to_dict()
            print(f"  {split}: {empty} rows without {column}, first {example}")
    print(f"{split:<8}{len(out):>6} rows   " + "   ".join(report))

    if "caption" in merged.columns:
        check = frame[KEY + ["caption"]].merge(
            merged[KEY + ["caption"]].drop_duplicates(KEY),
            on=KEY, how="inner", suffixes=("_ours", "_theirs"))
        differ = (check["caption_ours"].str.strip()
                  != check["caption_theirs"].str.strip()).sum()
        if differ:
            print(f"  {split}: WARNING {differ} rows where the English caption "
                  f"differs from ours; the join key still matched")

    updated[split] = out

incomplete = {s: [c for c in columns if f[c].isna().any()] for s, f in updated.items()}
incomplete = {s: c for s, c in incomplete.items() if c}
if incomplete:
    print(f"\nnot writing: columns still incomplete {incomplete}")
    raise SystemExit(1)

if not args.write:
    print("\nnothing written; rerun with --write to overwrite the split files")
    raise SystemExit(0)

for split, frame in updated.items():
    path = languages.split_path(split)
    frame.to_csv(path, index=False)
    print(f"wrote {os.path.relpath(path, languages.REPO)}")

print("\nnow declare the new columns in src/languages.py, then run "
      "python src/languages.py to check them")
