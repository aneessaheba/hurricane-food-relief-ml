"""
Supervised modeling utilities: preprocessor, 5 classifiers, 2 regressors,
CV scoring with StratifiedGroupKFold (disaster_number as groups), plus
GridSearchCV for RF and Optuna for XGBoost.

CRITICAL: SMOTE is applied via imblearn.pipeline.Pipeline — never
sklearn's Pipeline (which would leak SMOTE into predict()).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC

from config import (
    RANDOM_STATE, CONTINUOUS_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES,
)


# -----------------------------------------------------------------------------
# Preprocessor
# -----------------------------------------------------------------------------
def build_preprocessor(
    continuous=None, categorical=None, binary=None,
) -> ColumnTransformer:
    continuous = continuous or CONTINUOUS_FEATURES
    categorical = categorical or CATEGORICAL_FEATURES
    binary = binary or BINARY_FEATURES
    return ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), continuous),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
             categorical),
            ("bin", "passthrough", binary),
        ],
        remainder="drop",
    )


# -----------------------------------------------------------------------------
# Classifier zoo
# -----------------------------------------------------------------------------
def get_classifiers() -> Dict[str, object]:
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            objective="multi:softprob", n_estimators=200, max_depth=6,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=RANDOM_STATE,
            n_jobs=-1, tree_method="hist",
        )
    except ImportError:
        xgb = None
    clfs = {
        "rf": RandomForestClassifier(
            n_estimators=200, max_features="sqrt", class_weight="balanced",
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        # RBF SVC is O(n^2) and impractical after SMOTE inflates the train set.
        # Use LinearSVC (still an SVM) wrapped in CalibratedClassifierCV so
        # predict_proba is available for the downstream priority index.
        "svm": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", dual="auto",
                      max_iter=5000, random_state=RANDOM_STATE),
            method="sigmoid", cv=3,
        ),
        "nb": GaussianNB(),
        # saga converges much faster than lbfgs on this (post-OHE sparse-ish) matrix.
        "lr": LogisticRegression(
            solver="saga", penalty="l2", C=1.0,
            max_iter=5000, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }
    if xgb is not None:
        clfs["xgb"] = xgb
    return clfs


def get_regressors() -> Dict[str, object]:
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
        )
    except ImportError:
        xgb = None
    reg = {
        "rf": RandomForestRegressor(
            n_estimators=400, max_features="sqrt",
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }
    if xgb is not None:
        reg["xgb"] = xgb
    return reg


# -----------------------------------------------------------------------------
# Pipeline constructor
# -----------------------------------------------------------------------------
def build_pipeline(model, preprocessor=None, use_smote: bool = True) -> ImbPipeline:
    pre = preprocessor or build_preprocessor()
    steps = [("pre", pre)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return ImbPipeline(steps)


# -----------------------------------------------------------------------------
# Cross-validation
# -----------------------------------------------------------------------------
def cv_score(
    pipeline: ImbPipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
    n_splits: int = 5, scoring: str = "f1_weighted",
) -> Tuple[float, np.ndarray]:
    """
    StratifiedGroupKFold with groups=disaster_number. Groups are passed to
    .split(), not the constructor.
    """
    n_groups = pd.Series(groups).nunique()
    n_splits = max(2, min(n_splits, n_groups))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                               random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in skf.split(X, y, groups=groups):
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipeline.predict(X.iloc[val_idx])
        if scoring == "f1_weighted":
            scores.append(f1_score(y.iloc[val_idx], pred, average="weighted"))
        else:
            raise ValueError(scoring)
    arr = np.array(scores)
    return arr.mean(), arr


# -----------------------------------------------------------------------------
# Tuning
# -----------------------------------------------------------------------------
def tune_rf_grid(X, y, groups, preprocessor=None):
    pipe = build_pipeline(RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    ), preprocessor=preprocessor)
    grid = {
        "model__n_estimators": [100, 200, 400],
        "model__max_depth":    [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
    }
    n_splits = max(2, min(5, pd.Series(groups).nunique()))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1)
    gs.fit(X, y, groups=groups)
    return gs


def tune_xgb_optuna(X, y, groups, n_trials: int = 50, preprocessor=None):
    import optuna
    from xgboost import XGBClassifier

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth":    trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":    trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1, tree_method="hist", **params,
        )
        pipe = build_pipeline(model, preprocessor=preprocessor)
        mean, _ = cv_score(pipe, X, y, groups)
        return mean

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


# -----------------------------------------------------------------------------
# Regression evaluation
# -----------------------------------------------------------------------------
def regression_eval(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def classification_eval(y_true, y_pred) -> Dict[str, object]:
    return {
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
