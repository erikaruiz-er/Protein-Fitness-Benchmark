"""
Evaluation harness: fit a simple regressor on top of whichever feature set
you give it, and score with Spearman rank correlation -- the standard
metric in this field (ProteinGym, FLIP, TAPE all report it), because what
matters for downstream use (picking the best variant to test in the lab)
is getting the *ranking* right, not the exact score.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .data import Assay
from .features import featurize_assay


@dataclass
class AssayResult:
    assay_name: str
    n_mutants: int
    spearman: float
    spearman_pvalue: float


def evaluate_assay(
    assay: Assay,
    featurizer=featurize_assay,
    n_folds: int = 5,
    seed: int = 0,
) -> AssayResult:
    """Cross-validated Spearman correlation for one assay.

    We use k-fold CV (rather than a single train/test split) because DMS
    assays are often only a few hundred to a few thousand mutants -- a
    single split gives a noisy estimate.
    """
    X = featurizer(assay.wildtype_sequence, assay.df["mutated_sequence"])
    y = assay.df["DMS_score"].to_numpy()

    kf = KFold(n_splits=min(n_folds, len(y)), shuffle=True, random_state=seed)
    preds = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in kf.split(X):
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])

    rho, pval = spearmanr(preds, y)
    return AssayResult(
        assay_name=assay.name,
        n_mutants=len(y),
        spearman=rho,
        spearman_pvalue=pval,
    )


def evaluate_many(assays: list[Assay], **kwargs) -> pd.DataFrame:
    """Run evaluate_assay over a list of assays and return a summary table,
    mirroring the per-assay + aggregate structure ProteinGym reports."""
    results = [evaluate_assay(a, **kwargs) for a in assays]
    df = pd.DataFrame([r.__dict__ for r in results])
    df.loc["mean"] = {
        "assay_name": "MEAN",
        "n_mutants": df["n_mutants"].sum(),
        "spearman": df["spearman"].mean(),
        "spearman_pvalue": np.nan,
    }
    return df
