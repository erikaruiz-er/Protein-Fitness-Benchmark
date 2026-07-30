import numpy as np

from src.data import make_synthetic_assay
from src.evaluate import evaluate_assay
from src.features import amino_acid_composition, mutation_features


def test_synthetic_assay_shape():
    assay = make_synthetic_assay(n_mutants=200, seed=1)
    assert len(assay.df) == 200
    assert set(["mutant", "mutated_sequence", "DMS_score"]) <= set(assay.df.columns)


def test_amino_acid_composition_sums_to_one():
    comp = amino_acid_composition("ACDEFGHIKLMNPQRSTVWY")
    assert np.isclose(comp.sum(), 1.0)


def test_mutation_features_detects_single_mutation():
    wt = "AAAA"
    mut = "AAPA"
    feats = mutation_features(wt, mut)
    assert feats[0] == 1  # n_mutations


def test_baseline_beats_random_on_synthetic_signal():
    """The synthetic assay has a known, learnable signal (core hydrophobic
    positions + one catalytic position). A reasonable baseline should
    recover a meaningfully positive Spearman correlation -- this is the
    regression test that would catch a broken featurizer or a data leak."""
    assay = make_synthetic_assay(n_mutants=400, seed=2)
    result = evaluate_assay(assay, n_folds=5, seed=2)
    assert result.spearman > 0.3, f"Expected clear signal, got rho={result.spearman:.3f}"
