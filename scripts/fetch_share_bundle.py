"""
Download + extract the team share bundle into the project tree.

Run from project root, AFTER `git clone`:
    python scripts/fetch_share_bundle.py <google_drive_share_url>

The URL can be:
    https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
    https://drive.google.com/open?id=<FILE_ID>
    https://drive.google.com/uc?id=<FILE_ID>
    or any direct https URL pointing to a zip (Dropbox, OneDrive, etc.)

After extraction the project tree will contain:
    data/processed/  data/interim/  models/  outputs/priority_rankings.csv
    (plus data/raw/ if the bundle was --full)
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract_drive_id(url: str) -> str | None:
    """Pull the Google Drive file id out of any of the common URL shapes."""
    for pat in (r"/file/d/([A-Za-z0-9_\-]+)",
                r"[?&]id=([A-Za-z0-9_\-]+)"):
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def download_via_gdown(url_or_id: str, dest: Path) -> bool:
    """Try the gdown package (handles Drive's confirm-token dance)."""
    try:
        import gdown  # type: ignore
    except ImportError:
        print("  [info] gdown not installed; falling back to curl")
        print("         pip install gdown  for the smoothest Drive support")
        return False
    print(f"  [gdown] downloading -> {dest.name}")
    try:
        # gdown.download accepts both ids and full sharing URLs
        result = gdown.download(url_or_id, str(dest), quiet=False, fuzzy=True)
        return bool(result) and dest.exists() and dest.stat().st_size > 1024
    except Exception as e:
        print(f"  [gdown] failed: {e}")
        return False


def download_via_curl(url: str, dest: Path) -> bool:
    """Fallback: plain curl. Works for direct URLs (Dropbox, OneDrive direct)
    and for Drive small files."""
    print(f"  [curl] downloading -> {dest.name}")
    try:
        subprocess.check_call([
            "curl", "-L", "-A", "Mozilla/5.0",
            "-o", str(dest), url,
        ])
        return dest.exists() and dest.stat().st_size > 1024
    except Exception as e:
        print(f"  [curl] failed: {e}")
        return False


def extract_zip(zip_path: Path, target: Path) -> int:
    print(f"  extracting {zip_path.name} -> {target}/ ...")
    with zipfile.ZipFile(zip_path) as z:
        members = z.namelist()
        z.extractall(target)
    return len(members)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="Google Drive share URL (or any direct zip URL)")
    ap.add_argument("--keep-zip", action="store_true",
                    help="Keep the downloaded zip after extraction (default: delete)")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "share_bundle.zip"

        # Strategy 1: Drive URL → use gdown
        drive_id = extract_drive_id(args.url)
        ok = False
        if drive_id:
            ok = download_via_gdown(drive_id, zip_path)

        # Strategy 2: any URL → curl fallback
        if not ok:
            ok = download_via_curl(args.url, zip_path)

        if not ok:
            print("\n[error] download failed. Common fixes:")
            print("  - Make the Drive file 'Anyone with the link can view'")
            print("  - Install gdown:  pip install gdown")
            print("  - Manually download to the project root and unzip there")
            return 1

        n = extract_zip(zip_path, ROOT)
        print(f"  extracted {n} entries\n")

        if args.keep_zip:
            kept = ROOT / "share_bundle.zip"
            shutil.copy(zip_path, kept)
            print(f"  kept zip at: {kept}")

    # Verify a few key files landed
    print("verification:")
    checks = [
        ("data/processed/abt.csv",                    "analytic base table"),
        ("data/processed/abt_with_clusters.csv",      "ABT + cluster labels"),
        ("models/best_classifier.pkl",                "trained classifier"),
        ("models/best_regressor.pkl",                 "trained regressor"),
        ("outputs/priority_rankings.csv",             "priority rankings"),
    ]
    all_ok = True
    for rel, label in checks:
        p = ROOT / rel
        ok = p.exists() and p.stat().st_size > 0
        all_ok &= ok
        mark = "OK   " if ok else "MISS "
        size = f"{p.stat().st_size/1e6:>7.1f} MB" if ok else "      —"
        print(f"  [{mark}] {size}  {rel}    ({label})")

    if all_ok:
        print("\nAll set. You can now:")
        print("  - jupyter lab           (run notebooks 02-08 without re-downloading)")
        print("  - streamlit run app/streamlit_app.py")
        return 0
    print("\n[warn] some artifacts missing — bundle may have been --full vs partial")
    return 0


if __name__ == "__main__":
    sys.exit(main())
