"""
Rebuild every table and figure in the IEEE BigData 2026 paper (BigD261) and
check the numbers against the committed results.

    python scripts/reproduce_paper.py                  # all stages
    python scripts/reproduce_paper.py main loho        # selected stages
    python scripts/reproduce_paper.py --fetch          # download missing inputs first
    python scripts/reproduce_paper.py --check-only     # compare without rerunning

Steps: (1) check that the inputs exist, (2) check that data/processed/abt.csv
is byte-for-byte the published ABT (version 1; the corrected table from
src.pipeline is validated separately), (3) snapshot the
committed outputs/camera_ready/results.json, (4) run
scripts/camera_ready_experiments.py, (5) compare every number with the
snapshot. The paper reports at most three decimals, so values that differ by
less than --tol (default 5e-4) count as reproduced; Random Forest's parallel
training jitters some metrics around the sixth decimal.
Exits non-zero if the ABT is not the published one, validation of the
corrected table fails, or any number differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from config import DATA_PATHS  # noqa: E402
from src.validation import assert_valid  # noqa: E402

RESULTS = DATA_PATHS["outputs"] / "camera_ready" / "results.json"
ABT = DATA_PATHS["processed"] / "abt.csv"
ABT_CLEAN = DATA_PATHS["processed"] / "abt_clean.csv"
# SHA-256 of the published ABT (version 1) behind every number in the paper.
PUBLISHED_ABT_SHA256 = "88665ebeed431549b8f4cea88b908241f278872f9c3823b49ad5dd8dce194322"
RAW = DATA_PATHS["raw"]

# Inputs each experiment stage reads beyond the ABT.
STAGE_INPUTS = {
    "frpi": [RAW / "ihp_foodneed_by_zip.csv"],
    "forecast": [RAW / "ibtracs_na.csv", RAW / "zcta_gazetteer_2020.txt"],
    "acs": [RAW / f"acs5_{y}_zcta.csv" for y in (2016, 2017, 2018, 2020, 2021, 2022, 2023, 2024)],
    "cleaned": [ABT_CLEAN],
}
ALL_STAGES = ["main", "binning", "selection", "calibration", "formulation", "loho",
              "zip", "ablation", "frpi", "forecast", "acs", "cleaned", "figures"]


def _leaves(obj, path: str = "") -> Iterator[Tuple[str, object]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, obj


def compare_results(reference: dict, new: dict, tol: float = 5e-4,
                    sections: List[str] | None = None) -> List[str]:
    """List every leaf in `reference` that is missing from `new` or differs."""
    ref_leaves = dict(_leaves(reference))
    new_leaves = dict(_leaves(new))
    problems = []
    for key, ref in ref_leaves.items():
        if sections and key.split(".", 1)[0].split("[", 1)[0] not in sections:
            continue
        if key not in new_leaves:
            problems.append(f"{key}: missing (reference {ref})")
            continue
        got = new_leaves[key]
        if isinstance(ref, (int, float)) and isinstance(got, (int, float)) \
                and not isinstance(ref, bool):
            if abs(float(ref) - float(got)) > tol:
                problems.append(f"{key}: {got} != reference {ref}")
        elif ref != got:
            problems.append(f"{key}: {got!r} != reference {ref!r}")
    return problems


def committed_reference() -> dict:
    """results.json as committed in git, or the file on disk if git is unavailable."""
    rel = RESULTS.relative_to(ROOT).as_posix()
    try:
        out = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
        return json.loads(out)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return json.loads(RESULTS.read_text())


def missing_inputs(stages: List[str]) -> List[Path]:
    needed = [ABT] + [p for s in stages for p in STAGE_INPUTS.get(s, [])]
    return [p for p in needed if not p.exists()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stages", nargs="*",
                    help=f"experiment stages to run (default: all): {' '.join(ALL_STAGES)}")
    ap.add_argument("--fetch", action="store_true", help="download missing inputs first")
    ap.add_argument("--check-only", action="store_true",
                    help="compare the current results.json with the committed one")
    ap.add_argument("--tol", type=float, default=5e-4)
    args = ap.parse_args(argv)
    unknown = sorted(set(args.stages) - set(ALL_STAGES))
    if unknown:
        ap.error(f"unknown stages {unknown}; choose from {' '.join(ALL_STAGES)}")
    stages = args.stages or ALL_STAGES
    py = sys.executable

    missing = missing_inputs(stages)
    if missing and args.fetch:
        subprocess.run([py, str(ROOT / "scripts" / "fetch_camera_ready_data.py")],
                       cwd=ROOT, check=True)
        missing = missing_inputs(stages)
    if missing:
        print("Missing inputs:\n  " + "\n  ".join(str(p.relative_to(ROOT)) for p in missing))
        print("Get the ABT from the share bundle (README), build abt_clean.csv with "
              "python -m src.pipeline --from features, and run with --fetch "
              "(ACS needs CENSUS_API_KEY).")
        return 2

    print("== checking the published ABT ==")
    digest = hashlib.sha256(ABT.read_bytes()).hexdigest()
    if digest != PUBLISHED_ABT_SHA256:
        print(f"{ABT.relative_to(ROOT)} is not the published ABT (sha256 {digest[:12]}...); "
              "restore it from the share bundle")
        return 1
    if ABT_CLEAN.exists():
        print("== validating the corrected ABT ==")
        try:
            for issue in assert_valid(pd.read_csv(ABT_CLEAN, dtype={"zip_code": str})):
                print(" ", issue)
        except ValueError as e:
            print(e)
            return 1

    reference = committed_reference()
    if not args.check_only:
        print(f"== running stages: {' '.join(stages)} ==")
        subprocess.run([py, str(ROOT / "scripts" / "camera_ready_experiments.py"), *stages],
                       cwd=ROOT, check=True)

    sections = None if args.check_only else [s for s in stages if s != "figures"]
    problems = compare_results(reference, json.loads(RESULTS.read_text()),
                               tol=args.tol, sections=sections)
    checked = sum(1 for k, _ in _leaves(reference)
                  if sections is None or k.split(".", 1)[0] in sections)
    if problems:
        print(f"\n{len(problems)} of {checked} reference values differ (tol {args.tol}):")
        print("  " + "\n  ".join(problems[:50]))
        return 1
    print(f"\nReproduced: all {checked} reference values match within {args.tol}.")
    print(f"Figures and tables: {RESULTS.parent.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
