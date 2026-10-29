import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "reproduce_paper", Path(__file__).resolve().parents[1] / "scripts" / "reproduce_paper.py")
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)

REF = {"main": {"auc": 0.7837, "labels": ["High", "Low"], "n": 2587},
       "loho": {"per_storm": [0.70, 0.81]}}


def test_identical_results_reproduce():
    assert reproduce.compare_results(REF, REF) == []


def test_float_jitter_below_tolerance_is_ignored():
    new = {"main": {"auc": 0.7837004, "labels": ["High", "Low"], "n": 2587},
           "loho": {"per_storm": [0.7000001, 0.81]}}
    assert reproduce.compare_results(REF, new) == []


def test_changed_and_missing_values_are_reported():
    new = {"main": {"auc": 0.79, "labels": ["High", "Low"]}, "loho": {"per_storm": [0.70, 0.81]}}
    problems = reproduce.compare_results(REF, new)
    assert any(p.startswith("main.auc") for p in problems)
    assert any(p.startswith("main.n: missing") for p in problems)
    assert len(problems) == 2


def test_non_numeric_values_compare_exactly():
    new = {**REF, "main": {**REF["main"], "labels": ["Low", "High"]}}
    assert reproduce.compare_results(REF, new) == [
        "main.labels[0]: 'Low' != reference 'High'",
        "main.labels[1]: 'High' != reference 'Low'"]


def test_sections_limit_the_comparison():
    new = {"main": REF["main"], "loho": {"per_storm": [0.5, 0.81]}}
    assert reproduce.compare_results(REF, new, sections=["main"]) == []
    assert len(reproduce.compare_results(REF, new, sections=["loho"])) == 1


def test_unknown_stage_is_rejected():
    with pytest.raises(SystemExit):
        reproduce.main(["not-a-stage"])
