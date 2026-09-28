import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TARGET = os.path.join(REPO, "data", "clip_embeddings")

BASE_URL = os.environ.get(
    "PQPP_ARTIFACTS_URL",
    "https://github.com/Adriana19Valentina/multilingual-pqpp/releases/download/v1.0",
)

BUNDLES = {
    "xlmr": {
        "file": "pqpp-precomputed-xlmr.tar.gz",
        "sha256": "c539afb87574a2d21f3e08abb49fdf9ae16bb3f0b245760a40cc3bcf9121ff28",
        "size_mb": 249,
        "contains": [
            "image_embeddings_xlmr-vitb32.npz",
            "retrieved_image_embeddings_xlmr-vitb32.npz",
            "retrieval_lists_clip.npz",
            "retrieval_lists_blip2.npz",
        ],
    },
    "longclip": {
        "file": "pqpp-precomputed-longclip.tar.gz",
        "sha256": "8ac492d44667a420fdffd0915d545b0d21c57763a3b67c9b2c6ec9e2611768c4",
        "size_mb": 169,
        "contains": [
            "image_embeddings_longclip-b.npz",
            "retrieved_image_embeddings_longclip-b.npz",
            "text_embeddings_longclip-b.npz",
        ],
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--bundle", nargs="*", default=["xlmr"], choices=sorted(BUNDLES),
                    help="which bundles to fetch; 'xlmr' is enough to run every "
                         "multilingual experiment, 'longclip' adds the control runs")
parser.add_argument("--keep-archive", action="store_true")
parser.add_argument("--check", action="store_true",
                    help="only report what is already present, download nothing")
args = parser.parse_args()

os.makedirs(TARGET, exist_ok=True)


def digest(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def present(bundle):
    return [name for name in bundle["contains"]
            if not os.path.exists(os.path.join(TARGET, name))]


if args.check:
    for name in sorted(BUNDLES):
        missing = present(BUNDLES[name])
        state = "complete" if not missing else f"{len(missing)} file(s) missing"
        print(f"{name:<10} {state}")
        for item in missing:
            print(f"           {item}")
    sys.exit(0)

for name in args.bundle:
    bundle = BUNDLES[name]
    missing = present(bundle)
    if not missing:
        print(f"{name}: already complete, nothing to do")
        continue

    url = f"{BASE_URL}/{bundle['file']}"
    archive = os.path.join(TARGET, bundle["file"])
    print(f"{name}: downloading {bundle['file']} (~{bundle['size_mb']} MB)")
    print(f"  from {url}")

    def progress(count, block_size, total):
        if total <= 0:
            return
        done = min(count * block_size, total)
        sys.stdout.write(f"\r  {done / 2**20:>6.0f} / {total / 2**20:.0f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, archive, reporthook=progress)
    except Exception as error:
        print(f"\n  download failed: {error}")
        print(f"  fetch {bundle['file']} manually and place it in {TARGET}, "
              f"then rerun this script")
        sys.exit(1)
    print()

    actual = digest(archive)
    if actual != bundle["sha256"]:
        print(f"  checksum mismatch\n    expected {bundle['sha256']}\n"
              f"    got      {actual}")
        print("  the archive is corrupt or the release was replaced; not extracting")
        sys.exit(1)
    print(f"  checksum ok")

    with tarfile.open(archive) as tar:
        tar.extractall(TARGET)
    if not args.keep_archive:
        os.remove(archive)

    still_missing = present(bundle)
    if still_missing:
        print(f"  WARNING: after extraction these are still missing: {still_missing}")
    else:
        print(f"  extracted into {os.path.relpath(TARGET, REPO)}")

print("\nWhat is still language-specific and must be computed locally:")
print("  python src/clip/compute_text_embeddings.py --encoder xlmr-vitb32")
