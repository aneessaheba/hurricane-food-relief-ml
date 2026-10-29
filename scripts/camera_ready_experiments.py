"""
Camera-ready experiments for IEEE BigData 2026 paper BigD261.

Retrains the deployment classifier exactly as the paper describes it (one
pipeline, one seed) and regenerates every reported number, plus the analyses
the reviewers asked for:

  main         Tables V-VIII, confusion matrix, Brier, SHAP, regression,
               FRPI validation (uses isotonic-calibrated probabilities)
  binning      alternative severity thresholds (Table IV)
  selection    leave-one-storm-out model selection on the 9 non-TEST storms (R2)
  calibration  isotonic calibration fit on VAL; XGBoost vs RF probabilities
  formulation  multiclass vs ordinal vs regression formulations (R3)
  loho         leave-one-hurricane-out, expanding-window, storm bootstrap (R1, R2)
  zip          seen vs unseen ZIPs, ZIP-disjoint retraining (R4)
  ablation     feature-subset ablation (R3)
  frpi         ranking error propagation, circularity, IHP foodNeed validation (R4)
  forecast     NHC-scale track/intensity forecast-error stress test (R3)
  acs          year-matched ACS vintages vs static 2021 (R1)
  cleaned      the same models on data/processed/abt_clean.csv, the ABT with
               mis-keyed ZIPs removed and Census codes cleared (src.pipeline)
  figures      plots built from results.json (forecast lead-time skill)

Usage:  python scripts/camera_ready_experiments.py [stage ...]   (default: all)
Inputs: data/processed/abt.csv (the published ABT), plus data/raw/* from
        fetch_camera_ready_data.py and, for `cleaned`, abt_clean.csv
Output: outputs/camera_ready/results.json, *.csv, *.png
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.frozen import FrozenEstimator  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, log_loss, mean_absolute_error,
    mean_squared_error, precision_recall_fscore_support, r2_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline as SkPipeline  # noqa: E402
from xgboost import XGBClassifier, XGBRegressor  # noqa: E402

from config import (  # noqa: E402
    CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, DATA_PATHS, FEATURE_GROUPS,
    HURRICANE_META, MODEL_BINARY_FEATURES, MODEL_FEATURES, RANDOM_STATE,
    SEVERITY_BINS, SEVERITY_LABELS, TARGET_CLASS_COL, TARGET_COL,
)
from src.feature_engineering import derive_demographic_shares  # noqa: E402
from src.modeling import (  # noqa: E402
    build_pipeline, build_preprocessor, get_classifiers, get_regressors,
)
from src.priority_index import apply_fragility, fit_fragility_scalers  # noqa: E402

warnings.filterwarnings("ignore")

RAW = DATA_PATHS["raw"]
OUT = DATA_PATHS["outputs"] / "camera_ready"
OUT.mkdir(parents=True, exist_ok=True)

CONT = list(CONTINUOUS_FEATURES)
BIN = list(MODEL_BINARY_FEATURES)
CAT = list(CATEGORICAL_FEATURES)
FEATURES = list(MODEL_FEATURES)

CLASSES = sorted(SEVERITY_LABELS)          # LabelEncoder order: High, Low, Medium, Severe
SEV = CLASSES.index("Severe")
HS = [CLASSES.index("High"), CLASSES.index("Severe")]
MODEL_NAMES = {"xgb": "XGBoost", "rf": "Random Forest", "svm": "LinearSVC (cal.)",
               "lr": "Logistic Reg.", "nb": "Gaussian NB"}
W_RISK, W_FRAG = 0.6, 0.4
YEAR = {h["disaster_number"]: h["year"] for h in HURRICANE_META}

RESULTS: dict = {}
_cache: dict = {}


# -----------------------------------------------------------------------------
# Data and model helpers
# -----------------------------------------------------------------------------
def load_abt(name: str = "abt.csv") -> pd.DataFrame:
    df = pd.read_csv(DATA_PATHS["processed"] / name, dtype={"zip_code": str})
    df = df.dropna(subset=[TARGET_CLASS_COL]).reset_index(drop=True)
    df["y"] = df[TARGET_CLASS_COL].map({c: i for i, c in enumerate(CLASSES)})
    return df


def split(df, name):
    return df[df["train_test_split"] == name]


def make_pipe(name="xgb", features=FEATURES):
    cont = [c for c in CONT if c in features]
    binf = [c for c in BIN if c in features]
    cat = [c for c in CAT if c in features]
    pre = build_preprocessor(continuous=cont, categorical=cat, binary=binf)
    return build_pipeline(clone(get_classifiers()[name]), preprocessor=pre)


def fit(train, name="xgb", features=FEATURES):
    return make_pipe(name, features).fit(train[features], train["y"])


def proba_full(model, X) -> np.ndarray:
    """predict_proba padded to all four classes (folds may miss a class)."""
    p = model.predict_proba(X)
    classes = list(model.classes_)
    if len(classes) == len(CLASSES):
        return p
    full = np.zeros((len(X), len(CLASSES)))
    for j, c in enumerate(classes):
        full[:, int(c)] = p[:, j]
    return full


def macro_auc(y, p) -> float:
    onehot = np.eye(len(CLASSES))[y]
    cols = [k for k in range(len(CLASSES)) if 0 < onehot[:, k].sum() < len(y)]
    return float(roc_auc_score(onehot[:, cols], p[:, cols], average="macro"))


def binary_auc(y_bin, score) -> float:
    y_bin = np.asarray(y_bin)
    return float(roc_auc_score(y_bin, score)) if 0 < y_bin.sum() < len(y_bin) else float("nan")


def brier_per_class(y, p) -> np.ndarray:
    onehot = np.eye(len(CLASSES))[y]
    return ((p - onehot) ** 2).mean(axis=0)


def ece(y_bin, prob, n_bins=10) -> float:
    y_bin, prob = np.asarray(y_bin, float), np.asarray(prob, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(prob[m].mean() - y_bin[m].mean())
    return float(total)


def clf_metrics(y, p) -> dict:
    y = np.asarray(y)
    pred = p.argmax(axis=1)
    sev_true, sev_pred = y == SEV, pred == SEV
    return {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, pred)),
        "f1_weighted": float(f1_score(y, pred, average="weighted")),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "auc_macro": macro_auc(y, p),
        "recall_severe": float(sev_pred[sev_true].mean()) if sev_true.any() else float("nan"),
        "precision_severe": float(sev_true[sev_pred].mean()) if sev_pred.any() else float("nan"),
        "auc_high_or_severe": binary_auc(np.isin(y, HS), p[:, HS].sum(axis=1)),
    }


def fmt(d: dict) -> str:
    return "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def calibrate(model, val):
    cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    return cal.fit(val[FEATURES], val["y"])


def main_model(df):
    """XGBoost fit on TRAIN plus its VAL-fit isotonic calibrator (cached)."""
    if "xgb" not in _cache:
        m = fit(split(df, "TRAIN"))
        _cache["xgb"] = (m, calibrate(m, split(df, "VAL")))
    return _cache["xgb"]


def fragility(df) -> pd.DataFrame:
    scalers = fit_fragility_scalers(split(df, "TRAIN"))
    return apply_fragility(df, scalers)


def frpi(p_hs, ffs, w_risk=W_RISK):
    return w_risk * np.asarray(p_hs) + (1 - w_risk) * np.asarray(ffs)


def top_bottom(te, score, k=50) -> dict:
    order = np.argsort(-np.asarray(score), kind="stable")
    top, bot = te.iloc[order[:k]], te.iloc[order[-k:]]
    bot_dmg = bot[TARGET_COL].mean()
    return {
        "top_food_desert_pct": float(100 * top["food_desert_flag"].mean()),
        "bottom_food_desert_pct": float(100 * bot["food_desert_flag"].mean()),
        "top_svi_pctile": float(100 * top["svi_overall"].mean()),
        "bottom_svi_pctile": float(100 * bot["svi_overall"].mean()),
        "top_damage_per_1k": float(top[TARGET_COL].mean()),
        "bottom_damage_per_1k": float(bot_dmg),
        "damage_ratio": float(top[TARGET_COL].mean() / bot_dmg) if bot_dmg > 0 else float("inf"),
        "top_true_high_or_severe_pct": float(100 * top["y"].isin(HS).mean()),
    }


def save_results():
    with open(OUT / "results.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)


# -----------------------------------------------------------------------------
# Stage: main
# -----------------------------------------------------------------------------
def stage_main(df):
    tr, va, te = split(df, "TRAIN"), split(df, "VAL"), split(df, "TEST")
    res = {"split_rows": {s: int(len(split(df, s))) for s in ["TRAIN", "VAL", "TEST"]},
           "train_class_counts": {CLASSES[k]: int(v) for k, v in tr["y"].value_counts().items()},
           "test_class_counts": {CLASSES[k]: int(v) for k, v in te["y"].value_counts().items()}}

    # Table V: five classifiers + dummies, selected on VAL macro AUC
    table = {}
    for name in ["xgb", "rf", "svm", "lr", "nb"]:
        m = main_model(df)[0] if name == "xgb" else fit(tr, name)
        table[MODEL_NAMES[name]] = {
            "val": clf_metrics(va["y"], proba_full(m, va[FEATURES])),
            "test": clf_metrics(te["y"], proba_full(m, te[FEATURES])),
        }
        print(f"{MODEL_NAMES[name]:18s} TEST {fmt(table[MODEL_NAMES[name]]['test'])}")
    for strat in ["stratified", "most_frequent"]:
        d = DummyClassifier(strategy=strat, random_state=RANDOM_STATE).fit(tr[FEATURES], tr["y"])
        table[f"Dummy ({strat})"] = {"test": clf_metrics(te["y"], proba_full(d, te[FEATURES]))}
    res["model_table"] = table
    res["selected_on_val_auc"] = max(MODEL_NAMES.values(), key=lambda k: table[k]["val"]["auc_macro"])

    # Table VI + confusion matrix + Brier for XGBoost
    m, cal = main_model(df)
    p = proba_full(m, te[FEATURES])
    pred = p.argmax(axis=1)
    prec, rec, f1, sup = precision_recall_fscore_support(te["y"], pred, labels=range(4), zero_division=0)
    per = {CLASSES[k]: {"precision": prec[k], "recall": rec[k], "f1": f1[k], "support": int(sup[k])}
           for k in range(4)}
    for avg in ["weighted", "macro"]:
        pa, ra, fa, _ = precision_recall_fscore_support(te["y"], pred, average=avg, zero_division=0)
        per[f"{avg}_avg"] = {"precision": pa, "recall": ra, "f1": fa, "support": int(len(te))}
    res["xgb_per_class"] = per
    cm = confusion_matrix(te["y"], pred, labels=range(4))
    res["confusion"] = {"labels": CLASSES, "matrix": cm.tolist()}
    high, sev = CLASSES.index("High"), CLASSES.index("Severe")
    res["high_predicted_severe"] = {"count": int(cm[high, sev]), "share_of_true_high": float(cm[high, sev] / cm[high].sum()),
                                    "share_of_predicted_severe": float(cm[high, sev] / cm[:, sev].sum())}
    res["severe_predicted_high_or_severe"] = float(cm[sev, HS].sum() / cm[sev].sum())
    res["predicted_severe_true_high_or_severe"] = float(cm[HS, sev].sum() / cm[:, sev].sum())

    order = [CLASSES.index(c) for c in SEVERITY_LABELS]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(cm[np.ix_(order, order)], cmap="Blues")
    for i in range(4):
        for j in range(4):
            v = cm[order[i], order[j]]
            ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks(range(4), SEVERITY_LABELS)
    ax.set_yticks(range(4), SEVERITY_LABELS)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "test_confusion.png", dpi=300)
    plt.close(fig)

    prior = np.tile(np.bincount(tr["y"], minlength=4) / len(tr), (len(te), 1))
    pc = proba_full(cal, te[FEATURES])
    res["brier"] = {
        "raw_per_class": dict(zip(CLASSES, brier_per_class(te["y"].to_numpy(), p))),
        "raw_mean": float(brier_per_class(te["y"].to_numpy(), p).mean()),
        "calibrated_per_class": dict(zip(CLASSES, brier_per_class(te["y"].to_numpy(), pc))),
        "calibrated_mean": float(brier_per_class(te["y"].to_numpy(), pc).mean()),
        "train_prior_mean": float(brier_per_class(te["y"].to_numpy(), prior).mean()),
        "uniform_mean": float(brier_per_class(te["y"].to_numpy(), np.full_like(p, 0.25)).mean()),
    }

    # Equity audit by SVI quartile (TEST)
    q = pd.qcut(te["svi_overall"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    eq = {}
    for g in ["Q1", "Q2", "Q3", "Q4"]:
        mask = (q == g).to_numpy()
        yy, pp = te["y"].to_numpy()[mask], pred[mask]
        eq[g] = {"n": int(mask.sum()), "acc": float(accuracy_score(yy, pp)),
                 "f1_weighted": float(f1_score(yy, pp, average="weighted")),
                 "recall_severe": float((pp[yy == SEV] == SEV).mean()),
                 "precision_severe": float((yy[pp == SEV] == SEV).mean()),
                 "true_severe_rate": float((yy == SEV).mean()),
                 "pred_severe_rate": float((pp == SEV).mean())}
    rates = [eq[g]["pred_severe_rate"] for g in eq]
    base = [eq[g]["true_severe_rate"] for g in eq]
    from fairlearn.metrics import equalized_odds_difference
    eq["dp_ratio"] = float(min(rates) / max(rates))
    eq["base_rate_ratio"] = float(min(base) / max(base))
    eq["eo_difference"] = float(equalized_odds_difference(te["y"] == SEV, pred == SEV, sensitive_features=q))
    eq["overall"] = {"acc": float(accuracy_score(te["y"], pred)),
                     "f1_weighted": float(f1_score(te["y"], pred, average="weighted")),
                     "recall_severe": float((pred[te["y"] == SEV] == SEV).mean())}
    res["equity"] = eq

    # SHAP (Severe class, TEST)
    import shap
    pre, model = m.named_steps["pre"], m.named_steps["model"]
    Xt = pre.transform(te[FEATURES])
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    sv = np.asarray(shap.TreeExplainer(model).shap_values(Xt))
    sv = sv[:, :, SEV] if sv.shape[-1] == len(CLASSES) else sv[SEV]
    mabs = np.abs(sv).mean(axis=0)
    order_f = np.argsort(-mabs)
    res["shap_top"] = [(names[i], float(mabs[i])) for i in order_f[:15]]
    shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=10, plot_size=(6, 3.8))
    plt.savefig(OUT / "shap_beeswarm_severe.png", dpi=300, bbox_inches="tight")
    plt.close("all")

    # Regression (exploratory): RF on log1p vs raw target, per TEST hurricane
    def reg(log):
        inner = SkPipeline([("pre", build_preprocessor(continuous=CONT, categorical=CAT, binary=BIN)),
                            ("reg", clone(get_regressors()["rf"]))])
        r = TransformedTargetRegressor(regressor=inner, func=np.log1p, inverse_func=np.expm1) if log else inner
        return r.fit(tr[FEATURES], tr[TARGET_COL])
    rlog, rraw = reg(True), reg(False)
    yhat, yhat_raw = rlog.predict(te[FEATURES]), rraw.predict(te[FEATURES])
    per_storm = {}
    for hn, idx in te.groupby("hurricane_name").groups.items():
        loc = te.index.get_indexer(idx)
        yt = te[TARGET_COL].to_numpy()[loc]
        per_storm[hn] = {"rmse": float(np.sqrt(mean_squared_error(yt, yhat[loc]))),
                         "mae": float(mean_absolute_error(yt, yhat[loc])),
                         "r2": float(r2_score(yt, yhat[loc]))}
    yt = te[TARGET_COL].to_numpy()
    res["regression"] = {
        "per_storm": per_storm,
        "log1p": {"rmse": float(np.sqrt(mean_squared_error(yt, yhat))), "mae": float(mean_absolute_error(yt, yhat))},
        "raw": {"rmse": float(np.sqrt(mean_squared_error(yt, yhat_raw))), "mae": float(mean_absolute_error(yt, yhat_raw))},
        "spearman_log1p": float(spearmanr(yt, yhat).statistic),
    }

    # FRPI (calibrated P(High u Severe)) + weight sensitivity + Medium-class coverage
    tef = fragility(df).loc[te.index]
    p_hs = pc[:, HS].sum(axis=1)
    sens = {}
    for w in [0.5, 0.6, 0.7, 1.0, 0.0]:
        sens[f"{w:.1f}/{1 - w:.1f}"] = top_bottom(tef, frpi(p_hs, tef["food_fragility_score"], w))
    res["frpi_weight_sensitivity"] = sens
    res["frpi_raw_probabilities"] = top_bottom(tef, frpi(p[:, HS].sum(axis=1), tef["food_fragility_score"]))
    score = frpi(p_hs, tef["food_fragility_score"])
    med = (te["y"] == CLASSES.index("Medium")).to_numpy()
    res["medium_class"] = {"n": int(med.sum()),
                           "frpi_ge_0.70": float((score[med] >= 0.70).mean()),
                           "frpi_ge_0.50": float((score[med] >= 0.50).mean()),
                           "frpi_quantile_0.70_all": float(np.quantile(score, 0.7))}
    res["frpi_range"] = [float(score.min()), float(score.max())]
    RESULTS["main"] = res
    print("XGB TEST per-class:", json.dumps({k: {kk: round(float(vv), 3) for kk, vv in v.items()} for k, v in per.items()}))
    print("equity:", json.dumps({k: v for k, v in eq.items() if not isinstance(v, dict)}, default=float))
    print("FRPI 0.6/0.4:", fmt(sens["0.6/0.4"]))


# -----------------------------------------------------------------------------
# Stage: binning - Table IV, alternative severity thresholds
# -----------------------------------------------------------------------------
def stage_binning(df):
    tr = split(df, "TRAIN")
    q = list(np.quantile(tr[TARGET_COL], [0.25, 0.5, 0.75]))
    schemes = {"Original (2/8/25)": [2, 8, 25], "A (3/10/30)": [3, 10, 30],
               "B (1/6/20)": [1, 6, 20], f"C (TRAIN quartiles {q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f})": q}
    out = {}
    for label, edges in schemes.items():
        d = df.copy()
        cls = pd.cut(d[TARGET_COL], [-np.inf, *edges, np.inf], labels=SEVERITY_LABELS).astype(str)
        d["y"] = cls.map({c: i for i, c in enumerate(CLASSES)})
        te = split(d, "TEST")
        out[label] = clf_metrics(te["y"].to_numpy(), proba_full(fit(split(d, "TRAIN")), te[FEATURES]))
        print(f"{label:34s} {fmt(out[label])}")
    RESULTS["binning"] = out


# -----------------------------------------------------------------------------
# Stage: selection (R2) - choose the model without touching TEST
# -----------------------------------------------------------------------------
def stage_selection(df):
    """Leave-one-storm-out over the 9 TRAIN+VAL storms, every classifier."""
    dev = df[df["train_test_split"] != "TEST"]
    rows = []
    for name in ["xgb", "rf", "svm", "lr", "nb"]:
        for dn, g in dev.groupby("disaster_number"):
            m = fit(dev[dev["disaster_number"] != dn], name)
            p = proba_full(m, g[FEATURES])
            rows.append({"model": MODEL_NAMES[name], "storm": g["hurricane_name"].iat[0],
                         "log_loss": float(log_loss(g["y"], np.clip(p, 1e-6, 1), labels=range(4))),
                         **clf_metrics(g["y"].to_numpy(), p)})
        print(f"  {MODEL_NAMES[name]} done")
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "selection_loso_dev.csv", index=False)
    cols = ["auc_macro", "auc_high_or_severe", "recall_severe", "f1_weighted", "log_loss"]
    summary = tab.groupby("model")[cols].agg(["mean", "std"])
    print(summary.round(3).to_string())
    RESULTS["selection"] = {m: {c: {"mean": float(summary.loc[m, (c, "mean")]), "std": float(summary.loc[m, (c, "std")])}
                                for c in cols} for m in summary.index}


# -----------------------------------------------------------------------------
# Stage: calibration (R2)
# -----------------------------------------------------------------------------
def stage_calibration(df):
    va, te = split(df, "VAL"), split(df, "TEST")
    y = te["y"].to_numpy()
    hs_true = np.isin(y, HS)
    out, curves = {}, {}
    for name in ["xgb", "rf"]:
        m = main_model(df)[0] if name == "xgb" else fit(split(df, "TRAIN"), "rf")
        cal = main_model(df)[1] if name == "xgb" else calibrate(m, va)
        for tag, model in [("raw", m), ("isotonic", cal)]:
            p = proba_full(model, te[FEATURES])
            out[f"{name}_{tag}"] = {
                "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1), labels=range(4))),
                "brier_mean": float(brier_per_class(y, p).mean()),
                "ece_severe": ece(y == SEV, p[:, SEV]),
                "ece_high_or_severe": ece(hs_true, p[:, HS].sum(axis=1)),
                "auc_macro": macro_auc(y, p),
                "auc_high_or_severe": binary_auc(hs_true, p[:, HS].sum(axis=1)),
                "recall_severe_argmax": float((p.argmax(1)[y == SEV] == SEV).mean()),
            }
            curves[f"{name}_{tag}"] = p[:, SEV]
        print(f"{name}: raw {fmt(out[name + '_raw'])}\n     iso {fmt(out[name + '_isotonic'])}")
    RESULTS["calibration"] = out

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    edges = np.linspace(0, 1, 11)
    for key, style, label in [("xgb_raw", "o-", "XGBoost (raw)"), ("xgb_isotonic", "s-", "XGBoost (isotonic)"),
                              ("rf_raw", "^--", "RF (raw)")]:
        pr = curves[key]
        idx = np.clip(np.digitize(pr, edges[1:-1]), 0, 9)
        xs = [pr[idx == b].mean() for b in range(10) if (idx == b).sum() >= 10]
        ys = [(y[idx == b] == SEV).mean() for b in range(10) if (idx == b).sum() >= 10]
        ax.plot(xs, ys, style, ms=3, lw=1, label=label)
    ax.plot([0, 1], [0, 1], ":", c="gray", lw=1)
    ax.set_xlabel("Predicted P(Severe)")
    ax.set_ylabel("Observed Severe fraction")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "calibration_severe.png", dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Stage: formulation (R3) - multiclass vs ordinal vs regression
# -----------------------------------------------------------------------------
def stage_formulation(df):
    tr, te = split(df, "TRAIN"), split(df, "TEST")
    y = te["y"].to_numpy()
    ordinal_rank = {CLASSES.index(c): i for i, c in enumerate(SEVERITY_LABELS)}  # Low=0..Severe=3
    yo_tr = tr["y"].map(ordinal_rank).to_numpy()
    out = {}

    p = proba_full(main_model(df)[0], te[FEATURES])
    exp_rank = p[:, [CLASSES.index(c) for c in SEVERITY_LABELS]] @ np.arange(4)
    out["multiclass_xgb"] = {**clf_metrics(y, p), "spearman_damage": float(spearmanr(te[TARGET_COL], exp_rank).statistic)}

    # Ordinal (Frank & Hall): P(y > k) from three binary XGBoost models
    pre = build_preprocessor(continuous=CONT, categorical=CAT, binary=BIN).fit(tr[FEATURES])
    Xtr, Xte = pre.transform(tr[FEATURES]), pre.transform(te[FEATURES])
    gt = []
    for k in range(3):
        b = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
        gt.append(b.fit(Xtr, (yo_tr > k).astype(int)).predict_proba(Xte)[:, 1])
    gt = np.maximum.accumulate(np.column_stack(gt)[:, ::-1], axis=1)[:, ::-1]  # enforce monotone
    po = np.column_stack([1 - gt[:, 0], gt[:, 0] - gt[:, 1], gt[:, 1] - gt[:, 2], gt[:, 2]])
    p_ord = np.zeros_like(po)
    for i, c in enumerate(SEVERITY_LABELS):
        p_ord[:, CLASSES.index(c)] = po[:, i]
    out["ordinal_xgb"] = {**clf_metrics(y, p_ord), "spearman_damage": float(spearmanr(te[TARGET_COL], gt.sum(1)).statistic)}

    # Regression on log1p(damage), then binned with the same thresholds
    r = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                     random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
    yhat = np.expm1(r.fit(Xtr, np.log1p(tr[TARGET_COL])).predict(Xte))
    binned = pd.cut(yhat, bins=SEVERITY_BINS, labels=SEVERITY_LABELS).astype(str)
    pred = np.array([CLASSES.index(c) for c in binned])
    out["regression_xgb_binned"] = {
        "acc": float(accuracy_score(y, pred)), "f1_weighted": float(f1_score(y, pred, average="weighted")),
        "recall_severe": float((pred[y == SEV] == SEV).mean()),
        "precision_severe": float((y[pred == SEV] == SEV).mean()) if (pred == SEV).any() else float("nan"),
        "auc_high_or_severe": binary_auc(np.isin(y, HS), yhat),
        "spearman_damage": float(spearmanr(te[TARGET_COL], yhat).statistic),
    }
    RESULTS["formulation"] = out
    for k, v in out.items():
        print(f"{k:24s} {fmt(v)}")


# -----------------------------------------------------------------------------
# Stage: loho (R1, R2) - storm-level uncertainty
# -----------------------------------------------------------------------------
def stage_loho(df):
    rows = []
    for dn, g in df.groupby("disaster_number"):
        rest = df[df["disaster_number"] != dn]
        m = fit(rest)
        rows.append({"scheme": "LOHO", "storm": g["hurricane_name"].iat[0], "year": YEAR[dn],
                     "split": g["train_test_split"].iat[0], "n_train": len(rest),
                     **clf_metrics(g["y"].to_numpy(), proba_full(m, g[FEATURES]))})
        earlier = df[df["disaster_number"].map(YEAR) < YEAR[dn]]
        if earlier["disaster_number"].nunique() >= 2:
            m = fit(earlier)
            rows.append({"scheme": "expanding", "storm": g["hurricane_name"].iat[0], "year": YEAR[dn],
                         "split": g["train_test_split"].iat[0], "n_train": len(earlier),
                         **clf_metrics(g["y"].to_numpy(), proba_full(m, g[FEATURES]))})
        print(f"  {rows[-1]['storm']:9s} done")
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "loho_per_storm.csv", index=False)
    cols = ["auc_macro", "recall_severe", "precision_severe", "f1_weighted", "acc", "auc_high_or_severe"]
    summary = {}
    for scheme, g in tab.groupby("scheme"):
        summary[scheme] = {c: {"mean": float(g[c].mean()), "std": float(g[c].std()), "min": float(g[c].min()),
                               "max": float(g[c].max())} for c in cols}
        summary[scheme]["n_storms"] = int(len(g))

    # Storm-level (cluster) bootstrap of the fixed TEST evaluation
    te = split(df, "TEST")
    p = proba_full(main_model(df)[0], te[FEATURES])
    storms = te["disaster_number"].unique()
    rng = np.random.default_rng(RANDOM_STATE)
    boot = []
    for _ in range(1000):
        pick = rng.choice(storms, size=len(storms), replace=True)
        loc = np.concatenate([np.flatnonzero(te["disaster_number"].to_numpy() == s) for s in pick])
        mt = clf_metrics(te["y"].to_numpy()[loc], p[loc])
        boot.append([mt["auc_macro"], mt["recall_severe"], mt["f1_weighted"]])
    boot = np.array(boot)
    summary["test_storm_bootstrap_95ci"] = {
        k: [float(np.nanpercentile(boot[:, i], 2.5)), float(np.nanpercentile(boot[:, i], 97.5))]
        for i, k in enumerate(["auc_macro", "recall_severe", "f1_weighted"])}
    RESULTS["loho"] = summary
    print(tab[["scheme", "storm", "split", "n_train", "auc_macro", "recall_severe", "f1_weighted"]].round(3).to_string())
    print(json.dumps(summary["test_storm_bootstrap_95ci"]))


# -----------------------------------------------------------------------------
# Stage: zip (R4) - ZIP overlap between TRAIN/VAL and TEST
# -----------------------------------------------------------------------------
def stage_zip(df):
    tr, va, te = split(df, "TRAIN"), split(df, "VAL"), split(df, "TEST")
    seen_zips = set(tr["zip_code"]) | set(va["zip_code"])
    seen = te["zip_code"].isin(seen_zips).to_numpy()
    p = proba_full(main_model(df)[0], te[FEATURES])
    y = te["y"].to_numpy()
    out = {"test_rows": int(len(te)), "test_rows_seen_zip": int(seen.sum()),
           "test_unique_zips": int(te["zip_code"].nunique()),
           "test_unique_zips_seen": int(te.loc[seen, "zip_code"].nunique()),
           "seen": clf_metrics(y[seen], p[seen]), "unseen": clf_metrics(y[~seen], p[~seen])}

    # Memorisation baseline: repeat the ZIP's class from its latest TRAIN/VAL storm
    hist = pd.concat([tr, va]).assign(yr=lambda d: d["disaster_number"].map(YEAR))
    last = hist.sort_values("yr").groupby("zip_code")["y"].last()
    rep = te.loc[seen, "zip_code"].map(last).to_numpy()
    out["repeat_last_class_on_seen"] = {"acc": float(accuracy_score(y[seen], rep)),
                                        "recall_severe": float((rep[y[seen] == SEV] == SEV).mean())}

    # ZIP-disjoint retraining: drop every TRAIN row whose ZIP appears in TEST
    tr_dis = tr[~tr["zip_code"].isin(set(te["zip_code"]))]
    m = fit(tr_dis)
    out["zip_disjoint_train_rows"] = int(len(tr_dis))
    out["zip_disjoint"] = clf_metrics(y, proba_full(m, te[FEATURES]))
    # same-size random subsample control (isolates the effect of fewer rows)
    ctrl = []
    for s in range(5):
        sub = tr.sample(n=len(tr_dis), random_state=s)
        ctrl.append(clf_metrics(y, proba_full(fit(sub), te[FEATURES])))
    out["random_subsample_control"] = {k: float(np.mean([c[k] for c in ctrl])) for k in ctrl[0] if k != "n"}
    RESULTS["zip"] = out
    for k in ["seen", "unseen", "zip_disjoint", "random_subsample_control", "repeat_last_class_on_seen"]:
        print(f"{k:26s} {fmt(out[k])}")
    print(f"seen rows {out['test_rows_seen_zip']}/{out['test_rows']}, ZIP-disjoint train rows {len(tr_dis)}")


# -----------------------------------------------------------------------------
# Stage: ablation (R3) - feature subsets
# -----------------------------------------------------------------------------
def stage_ablation(df):
    tr, va, te = split(df, "TRAIN"), split(df, "VAL"), split(df, "TEST")
    storm = ["max_wind_speed_kt", "distance_to_track_km", "hurricane_category"]
    hazard = FEATURE_GROUPS["flood"]
    svi = FEATURE_GROUPS["svi"]
    demo = FEATURE_GROUPS["demographics"]
    food = ["snap_retailer_count", "snap_retailers_per_1k", "dist_nearest_supermarket_mi",
            "snap_participation_pct", "food_desert_flag"]

    # SHAP ranking on VAL (not TEST), aggregated back to raw features
    import shap
    m = main_model(df)[0]
    pre = m.named_steps["pre"]
    Xv = pre.transform(va[FEATURES])
    Xv = Xv.toarray() if hasattr(Xv, "toarray") else Xv
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    sv = np.asarray(shap.TreeExplainer(m.named_steps["model"]).shap_values(Xv))
    sv = sv[:, :, SEV] if sv.shape[-1] == len(CLASSES) else sv[SEV]
    imp = pd.Series(np.abs(sv).mean(0), index=names)
    raw_imp = imp.groupby(lambda n: next((f for f in CAT if n.startswith(f + "_")), n)).sum().sort_values(ascending=False)

    subsets = {
        "storm only": storm,
        "storm + hazard": storm + hazard,
        "storm + hazard + SVI": storm + hazard + svi,
        "storm + hazard + SVI + demographics": storm + hazard + svi + demo,
        "all except food access": [f for f in FEATURES if f not in food],
        "no storm features": [f for f in FEATURES if f not in storm],
        f"top-5 SHAP (VAL): {', '.join(raw_imp.index[:5])}": list(raw_imp.index[:5]),
        "top-10 SHAP (VAL)": list(raw_imp.index[:10]),
        "all features": FEATURES,
    }
    out = {}
    for label, feats in subsets.items():
        model = fit(tr, features=feats)
        out[label] = {"n_features": len(feats), **clf_metrics(te["y"].to_numpy(), proba_full(model, te[feats]))}
        print(f"{label[:40]:40s} {fmt(out[label])}")
    out["val_shap_ranking"] = list(raw_imp.index)
    RESULTS["ablation"] = out


# -----------------------------------------------------------------------------
# Stage: frpi (R4) - error propagation, circularity, foodNeed validation
# -----------------------------------------------------------------------------
def stage_frpi(df):
    te = split(df, "TEST")
    tef = fragility(df).loc[te.index].copy()
    _, cal = main_model(df)
    pc = proba_full(cal, te[FEATURES])
    tef["p_hs"] = pc[:, HS].sum(axis=1)
    tef["frpi"] = frpi(tef["p_hs"], tef["food_fragility_score"])
    tef["frpi_oracle"] = frpi(tef["y"].isin(HS).astype(float), tef["food_fragility_score"])

    fn_path = RAW / "ihp_foodneed_by_zip.csv"
    if fn_path.exists():
        fn = pd.read_csv(fn_path, dtype={"zip_code": str})
        tef = tef.merge(fn, on=["disaster_number", "zip_code"], how="left")
        tef["food_need_count"] = tef["food_need_count"].fillna(0)
        tef["food_need_per_1k"] = tef["food_need_count"] / tef["population"].replace(0, np.nan) * 1000
        # Share of registrants reporting food need: less tied to damage volume than per-capita counts
        tef["food_need_share"] = tef["food_need_count"] / tef["validRegistrations"].replace(0, np.nan)
    else:
        print("[warn] ihp_foodneed_by_zip.csv missing - run fetch_camera_ready_data.py ihp")

    def per_storm(fn_):
        vals = [fn_(g) for _, g in tef.groupby("disaster_number")]
        return {"mean": float(np.nanmean(vals)), "per_storm": [float(v) for v in vals]}

    def topk_overlap(a, b, frac=0.10):
        k = max(1, int(round(frac * len(a))))
        ta, tb = set(np.argsort(-a.to_numpy())[:k]), set(np.argsort(-b.to_numpy())[:k])
        return len(ta & tb) / k

    out = {"error_propagation": {
        "spearman_frpi_vs_oracle": per_storm(lambda g: spearmanr(g["frpi"], g["frpi_oracle"]).statistic),
        "top10pct_overlap_with_oracle": per_storm(lambda g: topk_overlap(g["frpi"], g["frpi_oracle"])),
    }}
    # What the top decile actually contains, per storm
    k_rows = []
    for _, g in tef.groupby("disaster_number"):
        k = max(1, int(round(0.10 * len(g))))
        top = g.nlargest(k, "frpi")
        k_rows.append({"true_low_share": (top["y"] == CLASSES.index("Low")).mean(),
                       "true_high_or_severe_share": top["y"].isin(HS).mean(),
                       "base_high_or_severe_share": g["y"].isin(HS).mean(),
                       "severe_captured_in_top_quartile": (g.nlargest(max(1, len(g) // 4), "frpi")["y"] == SEV).sum()
                       / max(1, (g["y"] == SEV).sum())})
    out["error_propagation"]["top_decile_composition"] = pd.DataFrame(k_rows).mean().to_dict()

    # Circularity / component comparison: which score best tracks outcomes?
    scores = {"FRPI (0.6/0.4)": "frpi", "damage risk only P(H u S)": "p_hs",
              "fragility only (FFS)": "food_fragility_score", "SVI only": "svi_overall"}
    targets = {"verified_damage_per_1k": TARGET_COL}
    if "food_need_per_1k" in tef:
        targets["ihp_food_need_per_1k"] = "food_need_per_1k"
        targets["ihp_food_need_share"] = "food_need_share"
    comp = {}
    for sname, scol in scores.items():
        comp[sname] = {}
        for tname, tcol in targets.items():
            comp[sname][f"spearman_{tname}"] = per_storm(lambda g: spearmanr(g[scol], g[tcol], nan_policy="omit").statistic)
            comp[sname][f"top_decile_lift_{tname}"] = per_storm(
                lambda g: g.nlargest(max(1, int(round(0.1 * len(g)))), scol)[tcol].mean() / g[tcol].mean())
    out["component_comparison"] = comp
    if "food_need_per_1k" in tef:
        out["food_need_vs_damage_spearman"] = per_storm(lambda g: spearmanr(g["food_need_per_1k"], g[TARGET_COL]).statistic)
        out["mean_food_need_share"] = float(tef["food_need_share"].mean())
    RESULTS["frpi"] = out
    print(json.dumps(out, indent=1, default=float)[:4000])


# -----------------------------------------------------------------------------
# Stage: forecast (R3) - pre-landfall input uncertainty
# -----------------------------------------------------------------------------
# NHC official Atlantic forecast errors, simple mean of annual averages 2019-2023
# (NHC verification tables 1989-present_OFCL_ATL_annual_trk_errors.pdf and
# 1990-present_OFCL_ATL_annual_int_errors.pdf). Track: 36.3 / 62.8 / 87.1 n mi.
NMI_KM = 1.852
TRACK_ERR_KM = {"24h": 36.34 * NMI_KM, "48h": 62.84 * NMI_KM, "72h": 87.06 * NMI_KM}
INTENSITY_ERR_KT = {"24h": 7.1, "48h": 9.48, "72h": 10.92}


def _shift_track(lons, lats, offset_km, geod):
    """Displace every track vertex perpendicular to the local heading."""
    lons, lats = np.asarray(lons, float), np.asarray(lats, float)
    nxt = np.r_[np.arange(1, len(lons)), len(lons) - 1]
    prv = np.r_[0, np.arange(len(lons) - 1)]
    az, _, _ = geod.inv(lons[prv], lats[prv], lons[nxt], lats[nxt])
    lon2, lat2, _ = geod.fwd(lons, lats, az + 90.0, np.full(len(lons), offset_km * 1000.0))
    return lon2, lat2


def _dist_to_track(zlon, zlat, tlon, tlat, geod):
    """Geodesic distance from points to a densified polyline (min over vertices)."""
    seg_lon, seg_lat = [], []
    for i in range(len(tlon) - 1):
        pts = geod.npts(tlon[i], tlat[i], tlon[i + 1], tlat[i + 1], 10)
        seg_lon += [tlon[i]] + [p[0] for p in pts]
        seg_lat += [tlat[i]] + [p[1] for p in pts]
    seg_lon, seg_lat = np.array(seg_lon + [tlon[-1]]), np.array(seg_lat + [tlat[-1]])
    best = np.full(len(zlon), np.inf)
    for lo, la in zip(seg_lon, seg_lat):
        _, _, d = geod.inv(zlon, zlat, np.full(len(zlon), lo), np.full(len(zlon), la))
        best = np.minimum(best, d)
    return best / 1000.0


def stage_forecast(df):
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    gz_path, ib_path = RAW / "zcta_gazetteer_2020.txt", RAW / "ibtracs_na.csv"
    if not (gz_path.exists() and ib_path.exists()):
        print("[skip] forecast: run fetch_camera_ready_data.py gazetteer ibtracs")
        return
    gz = pd.read_csv(gz_path, sep="\t", dtype={"GEOID": str})
    gz.columns = [c.strip() for c in gz.columns]
    cent = gz.set_index("GEOID")[["INTPTLONG", "INTPTLAT"]]
    ib = pd.read_csv(ib_path, skiprows=[1], low_memory=False, usecols=["SEASON", "NAME", "ISO_TIME", "LAT", "LON"])
    ib["SEASON"] = pd.to_numeric(ib["SEASON"], errors="coerce")
    tracks = {}
    for h in HURRICANE_META:
        t = ib[(ib["NAME"].str.upper() == h["name"].upper()) & (ib["SEASON"] == h["year"])].sort_values("ISO_TIME")
        tracks[h["disaster_number"]] = (pd.to_numeric(t["LON"]).to_numpy(), pd.to_numeric(t["LAT"]).to_numpy())

    dfc = df[df["zip_code"].isin(cent.index)]
    tr, va, te = split(dfc, "TRAIN"), split(dfc, "VAL"), split(dfc, "TEST")
    tef = fragility(df).loc[te.index]

    def perturb(frame, offsets_km, wind_kt):
        """Copy of frame[FEATURES] with each storm's track shifted and wind offset."""
        X = frame[FEATURES].copy()
        X[["distance_to_track_km", "max_wind_speed_kt"]] = X[["distance_to_track_km", "max_wind_speed_kt"]].astype(float)
        for dn, g in frame.groupby("disaster_number"):
            tlon, tlat = tracks[dn]
            if offsets_km.get(dn, 0.0):
                tlon, tlat = _shift_track(tlon, tlat, offsets_km[dn], geod)
            c = cent.loc[g["zip_code"]]
            X.loc[g.index, "distance_to_track_km"] = _dist_to_track(
                c["INTPTLONG"].to_numpy(), c["INTPTLAT"].to_numpy(), tlon, tlat, geod)
            X.loc[g.index, "max_wind_speed_kt"] += wind_kt.get(dn, 0.0)
        return X

    def random_offsets(frame, lead, rng):
        sd_km = TRACK_ERR_KM[lead] * np.sqrt(np.pi / 2)   # half-normal mean = mean error
        sd_kt = INTENSITY_ERR_KT[lead] * np.sqrt(np.pi / 2)
        storms = frame["disaster_number"].unique()
        return ({dn: rng.normal(0, sd_km) for dn in storms}, {dn: rng.normal(0, sd_kt) for dn in storms})

    def overlap(a, b):
        vals = []
        for _, g in te.groupby("disaster_number"):
            k = max(1, int(round(0.10 * len(g))))
            vals.append(len(set(a.loc[g.index].nlargest(k).index) & set(b.loc[g.index].nlargest(k).index)) / k)
        return float(np.mean(vals))

    def score(m, cal, X):
        p, pc = proba_full(m, X), proba_full(cal, X)
        return clf_metrics(te["y"].to_numpy(), p), pd.Series(frpi(pc[:, HS].sum(1), tef["food_fragility_score"]), index=te.index)

    X0 = perturb(te, {}, {})
    out = {"n_test_rows_with_centroid": int(len(te)),
           "centroid_vs_abt_distance_spearman": float(spearmanr(X0["distance_to_track_km"], te["distance_to_track_km"]).statistic),
           "centroid_vs_abt_distance_median_abs_diff_km": float((X0["distance_to_track_km"] - te["distance_to_track_km"]).abs().median())}

    # Model A: trained on observed tracks (the paper's model). Model B: trained on
    # TRAIN copies whose tracks/winds carry 48 h-scale forecast error (augmentation).
    m_a = fit(tr)
    cal_a = calibrate(m_a, va)
    rng = np.random.default_rng(RANDOM_STATE + 1)
    aug = [tr[FEATURES].assign(y=tr["y"])]
    for _ in range(5):
        o, w = random_offsets(tr, "48h", rng)
        aug.append(perturb(tr, o, w).assign(y=tr["y"]))
    aug = pd.concat(aug, ignore_index=True)
    m_b = make_pipe().fit(aug[FEATURES], aug["y"])
    o, w = random_offsets(va, "48h", rng)
    cal_b = CalibratedClassifierCV(FrozenEstimator(m_b), method="isotonic").fit(perturb(va, o, w), va["y"])

    for tag, m, cal in [("observed_train", m_a, cal_a), ("augmented_train", m_b, cal_b)]:
        base, base_sc = score(m, cal, X0)
        res = {"observed_track": base}
        for lead, km in TRACK_ERR_KM.items():
            for side in [-1, 1]:
                met, sc = score(m, cal, perturb(te, {dn: side * km for dn in tracks}, {}))
                res[f"shift_{lead}_{'left' if side < 0 else 'right'}"] = {**met, "frpi_top10_overlap": overlap(sc, base_sc)}
        for i, lead in enumerate(TRACK_ERR_KM):
            rng_t = np.random.default_rng(RANDOM_STATE + 100 + i)   # identical draws for both models
            draws = []
            for _ in range(20):
                o, w = random_offsets(te, lead, rng_t)
                met, sc = score(m, cal, perturb(te, o, w))
                draws.append({**met, "frpi_top10_overlap": overlap(sc, base_sc)})
            dd = pd.DataFrame(draws)
            res[f"monte_carlo_{lead}"] = {c: {"mean": float(dd[c].mean()), "std": float(dd[c].std())}
                                          for c in ["auc_macro", "recall_severe", "f1_weighted",
                                                    "auc_high_or_severe", "frpi_top10_overlap"]}
            print(f"  {tag:15s} MC {lead}: " + "  ".join(
                f"{c}={v['mean']:.3f}±{v['std']:.3f}" for c, v in res[f"monte_carlo_{lead}"].items()))
        print(f"  {tag:15s} observed: {fmt(base)}")
        out[tag] = res

    # Track-agnostic floor: no storm-geometry information at all
    no_storm = [f for f in FEATURES if f not in ["distance_to_track_km", "max_wind_speed_kt", "hurricane_category"]]
    out["no_storm_features"] = clf_metrics(te["y"].to_numpy(), proba_full(fit(tr, features=no_storm), te[no_storm]))
    RESULTS["forecast"] = out


# -----------------------------------------------------------------------------
# Stage: acs (R1) - year-matched ACS vintages
# -----------------------------------------------------------------------------
def stage_acs(df):
    years = sorted(set(df["disaster_number"].map(YEAR)) | {2021})
    missing = [y for y in years if not (RAW / f"acs5_{y}_zcta.csv").exists()]
    if missing:
        print(f"[skip] acs: missing vintages {missing} - run fetch_camera_ready_data.py acs")
        return
    demo = FEATURE_GROUPS["demographics"]

    def shares(year):
        a = pd.read_csv(RAW / f"acs5_{year}_zcta.csv", dtype={"zip_code": str})
        a["zip_code"] = a["zip_code"].str.zfill(5)
        a = a[a["population"] > 0]
        a["median_income"] = a["median_income"].where(a["median_income"] > 0)  # -666666666 sentinel
        return derive_demographic_shares(a).set_index("zip_code")[demo]

    vint = {y: shares(y) for y in years}

    def rebuild(year_of_row):
        d = df.copy()
        filled = 0
        for i, (z, yr) in enumerate(zip(d["zip_code"], year_of_row)):
            src = vint[yr] if z in vint[yr].index else (vint[2021] if z in vint[2021].index else None)
            if src is None:
                continue
            filled += int(z in vint[yr].index)
            d.loc[d.index[i], demo] = src.loc[z].to_numpy()
        return d, filled

    static, _ = rebuild([2021] * len(df))
    matched, n_matched = rebuild(df["disaster_number"].map(YEAR).tolist())
    out = {"rows_with_year_matched_zcta": int(n_matched), "rows": int(len(df))}
    for label, d in [("static_2021", static), ("year_matched", matched)]:
        m = fit(split(d, "TRAIN"))
        te = split(d, "TEST")
        out[label] = clf_metrics(te["y"].to_numpy(), proba_full(m, te[FEATURES]))
        print(f"{label:14s} {fmt(out[label])}")
    drift = (matched.loc[df["train_test_split"] == "TRAIN", demo] - static.loc[df["train_test_split"] == "TRAIN", demo])
    out["train_feature_mean_abs_change"] = drift.abs().mean().to_dict()
    RESULTS["acs"] = out


def stage_figures(df):
    """Plots built from results.json (no retraining): forecast lead-time skill."""
    fc = RESULTS.get("forecast", {}).get("observed_train")
    if fc is None:
        print("[skip] figures: run the forecast stage first")
        return
    leads = [0, 24, 48, 72]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for key, label, marker in [("auc_macro", "Macro ROC-AUC", "o"), ("recall_severe", "Severe recall", "s")]:
        mean = [fc["observed_track"][key]] + [fc[f"monte_carlo_{h}h"][key]["mean"] for h in leads[1:]]
        std = [0.0] + [fc[f"monte_carlo_{h}h"][key]["std"] for h in leads[1:]]
        ax.errorbar(leads, mean, yerr=std, marker=marker, ms=4, lw=1.2, capsize=3, label=label)
    ax.axhline(RESULTS["forecast"]["no_storm_features"]["auc_macro"], ls="--", c="gray", lw=1,
               label="AUC, no storm features")
    ax.set_xticks(leads, ["Observed", "24 h", "48 h", "72 h"])
    ax.set_xlabel("Storm inputs (forecast lead time)")
    ax.set_ylabel("TEST score")
    ax.set_ylim(0.2, 0.9)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "forecast_lead_time.png", dpi=300)
    plt.close(fig)
    print("wrote", OUT / "forecast_lead_time.png")


# -----------------------------------------------------------------------------
# Stage: cleaned data
# -----------------------------------------------------------------------------
def stage_cleaned(_df):
    """Model selection and TEST metrics on the corrected ABT (abt_clean.csv)."""
    df = load_abt("abt_clean.csv")
    out = {"rows": {s: int((df["train_test_split"] == s).sum()) for s in ("TRAIN", "VAL", "TEST")}}
    dropped = DATA_PATHS["processed"] / "abt_dropped_rows.csv"
    if dropped.exists():
        out["rows_removed"] = int(len(pd.read_csv(dropped)))

    dev = df[df["train_test_split"] != "TEST"]
    sel = {}
    for name in ["xgb", "rf", "svm", "lr", "nb"]:
        aucs = [macro_auc(g["y"].to_numpy(),
                          proba_full(fit(dev[dev["disaster_number"] != dn], name), g[FEATURES]))
                for dn, g in dev.groupby("disaster_number")]
        sel[MODEL_NAMES[name]] = {"mean": float(np.mean(aucs)), "std": float(np.std(aucs, ddof=1))}
    out["selection_auc_macro"] = sel
    print("  LOSO dev AUC:", {k: round(v["mean"], 3) for k, v in sel.items()})

    tr, va, te = split(df, "TRAIN"), split(df, "VAL"), split(df, "TEST")
    out["test"] = {}
    for name in ["xgb", "lr"]:
        m = fit(tr, name)
        out["test"][MODEL_NAMES[name]] = clf_metrics(te["y"].to_numpy(), proba_full(m, te[FEATURES]))
        if name == "xgb":
            p_cal = calibrate(m, va).predict_proba(te[FEATURES])
            out["test"]["XGBoost"]["auc_high_or_severe_calibrated"] = binary_auc(
                te["y"].isin(HS), p_cal[:, HS].sum(axis=1))
        print(f"  {MODEL_NAMES[name]} TEST:", fmt(out["test"][MODEL_NAMES[name]]))
    RESULTS["cleaned"] = out


STAGES = {
    "main": stage_main, "binning": stage_binning, "selection": stage_selection, "calibration": stage_calibration, "formulation": stage_formulation,
    "loho": stage_loho, "zip": stage_zip, "ablation": stage_ablation, "frpi": stage_frpi,
    "forecast": stage_forecast, "acs": stage_acs, "cleaned": stage_cleaned,
    "figures": stage_figures,
}

if __name__ == "__main__":
    data = load_abt()
    if (OUT / "results.json").exists():
        RESULTS.update(json.loads((OUT / "results.json").read_text()))
    for stage in sys.argv[1:] or list(STAGES):
        print(f"\n===== {stage} =====")
        STAGES[stage](data)
        save_results()
    print(f"\nwrote {OUT / 'results.json'}")
