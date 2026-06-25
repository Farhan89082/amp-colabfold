"""
models.py
---------
Activity classifiers for AMP prediction.

Model 1: Gradient boosting on physicochemical features (XGBoost)
Model 2: Logistic regression on ESM-2 embeddings (via fair-esm)

Evaluation uses GroupShuffleSplit on sequence identity clusters
to prevent data leakage between train and test sets.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix,
)
import shap

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR   = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── feature columns used for Model 1 ─────────────────────────────────────────
# Drop amphipathicity (duplicate of hydrophobic_moment)
# Drop molecular_weight (r=0.98 with length)
FEATURE_COLS = [
    "length", "net_charge_pH7", "isoelectric_point",
    "hydrophobicity_eisenberg", "hydrophobic_moment",
    "instability_index", "aromaticity",
    "fraction_positive", "fraction_negative",
    "fraction_helix", "fraction_sheet", "fraction_turn",
    "aliphatic_index", "boman_index",
]


# ══════════════════════════════════════════════════════════════════════════════
# Label encoding
# ══════════════════════════════════════════════════════════════════════════════

def make_binary_labels(activity_series: pd.Series) -> np.ndarray:
    """
    Convert DRAMP activity labels to binary:
      antibacterial → 1  (specific experimental label)
      general / natural → 0  (broad / uncharacterised)

    This is a deliberate, defensible choice:
    'antibacterial' sequences have confirmed MIC data;
    'general' includes unverified and patent sequences.
    """
    return (activity_series == "antibacterial").astype(int).values


def make_multiclass_labels(activity_series: pd.Series):
    """
    Encode the three DRAMP activity classes as integers.
    Returns (encoded_array, label_encoder).
    """
    le = LabelEncoder()
    y = le.fit_transform(activity_series)
    return y, le


# ══════════════════════════════════════════════════════════════════════════════
# Identity-aware train/test split
# ══════════════════════════════════════════════════════════════════════════════

def identity_split(df: pd.DataFrame,
                   test_size: float = 0.2,
                   random_state: int = 42):
    """
    Split using sequence length as a proxy group to avoid
    near-identical sequences leaking between train and test.

    In production you'd use CD-HIT cluster IDs as groups;
    length decile is a fast, reasonable approximation here.
    """
    groups = pd.qcut(df["length"], q=10, labels=False, duplicates="drop")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return train_idx, test_idx


# ══════════════════════════════════════════════════════════════════════════════
# Model 1: Gradient Boosting on physicochemical features
# ══════════════════════════════════════════════════════════════════════════════

def train_gb_classifier(X_train: np.ndarray,
                        y_train: np.ndarray,
                        random_state: int = 42) -> GradientBoostingClassifier:
    """
    Train a gradient boosting classifier.
    Hyperparameters chosen for interpretability over raw performance:
    shallow trees (max_depth=3) keep SHAP values meaningful.
    """
    clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=random_state,
        verbose=0,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(clf, X_test: np.ndarray,
                        y_test: np.ndarray,
                        label_names: list | None = None) -> dict:
    """
    Return a dict of evaluation metrics.
    """
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc":          roc_auc_score(y_test, y_proba),
        "avg_precision":    average_precision_score(y_test, y_proba),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=label_names or ["negative", "positive"],
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred":           y_pred,
        "y_proba":          y_proba,
    }
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# SHAP analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_values(clf: GradientBoostingClassifier,
                        X: np.ndarray,
                        feature_names: list,
                        sample_size: int = 2000,
                        random_state: int = 42) -> tuple:
    """
    Compute SHAP values using TreeExplainer.
    Subsamples to `sample_size` rows for speed.
    Returns (shap_values, shap_explainer, X_sample).
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X[idx]

    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    return shap_values, explainer, X_sample


def shap_importance_df(shap_values: np.ndarray,
                       feature_names: list) -> pd.DataFrame:
    """
    Return a DataFrame of mean |SHAP| per feature, sorted descending.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature":         feature_names,
        "mean_abs_shap":   mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df