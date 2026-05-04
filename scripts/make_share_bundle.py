"""
Build a single zip bundle containing all artifacts that teammates need to
inspect / re-run / extend the project — but that are too large to commit
to git.

Run from project root:
    python scripts/make_share_bundle.py            # default: artifacts only (~25 MB)
    python scripts/make_share_bundle.py --full     # include data/raw too (~5+ GB)

Output:
    share_bundle.zip            (artifacts; default)
    share_bundle_full.zip       (--full)

After running, upload the zip to Google Drive (right-click → "Share" →
"Anyone with the link"), then share the URL with your team. They run:
    python scripts/fetch_share_bundle.py <gdrive_url>
to populate their local clone.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Always-include paths (small, ~25 MB total).
ARTIFACT_PATHS = [
    "data/processed",                  # abt.csv, abt_with_clusters.csv, abt.xlsx
    "data/interim",                    # fused_zip_hurricane.csv (intermediate)
    "models",                          # best_classifier.pkl, best_regressor.pkl, fragility_scalers.pkl
    "outputs/priority_rankings.csv",   # gitignored — needed for streamlit app
]

# Heavy paths (only with --full)
RAW_PATHS = [
    "data/raw",                        # ~5+ GB FEMA + Census + NRI raw downloads
]


def collect_files(paths: list[str], root: Path) -> list[Path]:
    files = []
    for rel in paths:
        p = root / rel
        if not p.exists():
            print(f"  [warn] missing: {rel}")
            continue
        if p.is_file():
            files.append(p)
        else:
            for f in p.rglob("*"):
                if f.is_file():
                    files.append(f)
    return files


def build_zip(zip_path: Path, files: list[Path], root: Path) -> None:
    print(f"writing {zip_path.name} ...")
    total_bytes = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as z:
        for i, f in enumerate(files, 1):
            arcname = f.relative_to(root).as_posix()
            z.write(f, arcname=arcname)
            total_bytes += f.stat().st_size
            if i % 50 == 0 or i == len(files):
                print(f"  [{i:>4}/{len(files)}]  {arcname}", flush=True)
    out_mb = zip_path.stat().st_size / 1e6
    raw_mb = total_bytes / 1e6
    print(f"\nbundle written: {zip_path}")
    print(f"  files:         {len(files):,}")
    print(f"  raw size:      {raw_mb:,.1f} MB")
    print(f"  zipped size:   {out_mb:,.1f} MB")
    print(f"  compression:   {(1 - out_mb / raw_mb) * 100:.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                    help="Include data/raw (~5+ GB). Default: artifacts only (~25 MB).")
    args = ap.parse_args()

    paths = list(ARTIFACT_PATHS)
    out_name = "share_bundle.zip"
    if args.full:
        paths += RAW_PATHS
        out_name = "share_bundle_full.zip"

    files = collect_files(paths, ROOT)
    if not files:
        print("[error] nothing to bundle — did you run notebooks 01-08?")
        return 1

    out_path = ROOT / out_name
    build_zip(out_path, files, ROOT)

    print()
    print("Next steps:")
    print(f"  1. Upload {out_name} to Google Drive")
    print('  2. Right-click -> "Share" -> "Anyone with the link" -> "Viewer"')
    print("  3. Copy the share URL")
    print('  4. Tell teammates to run:')
    print(f"       python scripts/fetch_share_bundle.py '<URL>'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
