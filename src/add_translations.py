import argparse
import glob
import os
import sys as _sys

import pandas as pd

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import languages

ID_KEY = ["caption_id", "source"]

parser = argparse.ArgumentParser(
    description="Merge translation columns into "
                "data/pqpp_multilingual_{train,val,test}.csv. One file is the "
                "normal case: the script works out which split each row belongs "
                "to, so row order does not matter, you do not need to split the "
                "file yourself, and you do not need to have translated every "
                "prompt.")
parser.add_argument("input", nargs="+",
                    help="CSV file(s) or glob pattern(s) holding the translations")
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
    incoming.append(frame)
    print(f"read {os.path.basename(path)}  {len(frame)} rows  "
          f"columns: {list(frame.columns)}")
merged = pd.concat(incoming, ignore_index=True)


def normalise(series):
    return (series.astype(str).str.strip().str.lower()
            .str.replace(r"\s+", " ", regex=True))


if all(k in merged.columns for k in ID_KEY):
    KEY = ID_KEY
    print(f"\njoining on {tuple(ID_KEY)}")
elif "caption" in merged.columns:
    KEY = ["_join"]
    merged["_join"] = normalise(merged["caption"])
    print("\ncaption_id/source absent; joining on the English caption instead "
          "(unique across all 10,200 prompts, compared case- and "
          "whitespace-insensitively)")
else:
    raise SystemExit(
        "the input needs either both 'caption_id' and 'source', or a 'caption' "
        "column holding the original English prompt")

duplicates = int(merged.duplicated(KEY).sum())
if duplicates:
    raise SystemExit(f"{duplicates} duplicate join keys across the inputs")

splits = {s: pd.read_csv(languages.split_path(s)) for s in languages.SPLIT_FILES}
known = set().union(*[set(f.columns) for f in splits.values()])

if args.columns:
    columns = args.columns
else:
    columns = [c for c in merged.columns
               if c.startswith("caption_") and c not in ID_KEY and c not in known]
if not columns:
    raise SystemExit("no new translation column found; pass --columns explicitly")
absent = [c for c in columns if c not in merged.columns]
if absent:
    raise SystemExit(f"requested column(s) absent from the inputs: {absent}")
print(f"columns to merge: {columns}\n")

take = merged[KEY + columns].drop_duplicates(KEY)
updated, matched_keys = {}, set()

for split, frame in splits.items():
    left = frame.copy()
    if KEY == ["_join"]:
        left["_join"] = normalise(left["caption"])

    out = left.merge(take, on=KEY, how="left", suffixes=("", "__incoming"))
    assert len(out) == len(frame), f"{split}: the merge changed the row count"
    matched_keys |= set(
        map(tuple, out.loc[out[columns[0]].notna(), KEY].to_numpy()))

    report = []
    for column in columns:
        empty = int(out[column].isna().sum())
        report.append(f"{column}={len(out) - empty}/{len(out)}")
    print(f"{split:<8}{len(out):>6} rows   " + "   ".join(report))

    if "caption" in merged.columns and KEY == ID_KEY:
        check = frame[ID_KEY + ["caption"]].merge(
            merged[ID_KEY + ["caption"]].drop_duplicates(ID_KEY),
            on=ID_KEY, how="inner", suffixes=("_ours", "_theirs"))
        differ = int((normalise(check["caption_ours"])
                      != normalise(check["caption_theirs"])).sum())
        if differ:
            print(f"  {split}: WARNING {differ} rows where the English caption "
                  f"differs from ours, though the id matched")

    updated[split] = out.drop(columns=["_join"], errors="ignore")

unused = len(take) - len(matched_keys)
if unused > 0:
    print(f"\n{unused} of the {len(take)} incoming rows matched no prompt "
          f"and were ignored")
    if KEY == ["_join"]:
        print("  with a caption join this usually means the English text was "
              "edited; add caption_id and source to the file to avoid it")

total = sum(len(f) for f in updated.values())
covered = sum(int(f[columns].notna().all(axis=1).sum()) for f in updated.values())
if covered == 0:
    raise SystemExit(
        "\nno prompt was matched at all; check that caption_id and source (or "
        "the English caption) in your file come from this benchmark")
if covered < total:
    print(f"\n{covered}/{total} prompts translated. You do not need the full "
          f"{total}: the untranslated ones are dropped from every language "
          f"alike, so the comparison stays row-for-row.")

if not args.write:
    print("\nnothing written; rerun with --write to overwrite the split files")
    raise SystemExit(0)

for split, frame in updated.items():
    path = languages.split_path(split)
    frame.to_csv(path, index=False)
    print(f"wrote {os.path.relpath(path, languages.REPO)}")

print("\nnow declare the new columns in src/languages.py, then run "
      "python src/languages.py to check them")
