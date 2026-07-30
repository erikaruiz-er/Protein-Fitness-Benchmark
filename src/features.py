"""
Baseline featurizers for protein fitness prediction.

These don't require a GPU, internet access, or a pretrained model -- they're
the "does the simplest reasonable thing beat random" baseline that any real
model (e.g. an ESM/AlphaFold-embedding model, or GPT-Rosalind itself) should
be compared against. A model that can't beat these on an easy assay isn't
learning much.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import AMINO_ACIDS

# Kyte-Doolittle hydrophobicity scale (standard, widely used physicochemical
# descriptor). Source: Kyte J, Doolittle RF. J Mol Biol. 1982.
HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

CHARGE = {aa: 0 for aa in AMINO_ACIDS}
CHARGE.update({"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5})

BULKY = set("FWYRK")  # large side chains


def amino_acid_composition(sequence: str) -> np.ndarray:
    """Fraction of each of the 20 amino acids in the sequence. 20-dim."""
    counts = np.array([sequence.count(aa) for aa in AMINO_ACIDS], dtype=float)
    return counts / max(len(sequence), 1)


def global_physicochemical_features(sequence: str) -> np.ndarray:
    """Sequence-level averages: hydrophobicity, charge, bulkiness. 3-dim."""
    n = len(sequence)
    hydro = sum(HYDROPHOBICITY.get(aa, 0.0) for aa in sequence) / n
    charge = sum(CHARGE.get(aa, 0.0) for aa in sequence) / n
    bulky = sum(1 for aa in sequence if aa in BULKY) / n
    return np.array([hydro, charge, bulky])


def mutation_features(wildtype: str, mutant_seq: str) -> np.ndarray:
    """Features describing *how* a mutant differs from wildtype at the
    mutated site(s) -- this is usually the strongest signal for DMS data,
    since most of the sequence is unchanged. 5-dim.

    Returns: [n_mutations, mean hydrophobicity change, mean charge change,
              mean bulkiness change, position spread (0 for single mutants)]
    """
    assert len(wildtype) == len(mutant_seq), "sequences must be aligned / same length"
    diffs = [i for i, (a, b) in enumerate(zip(wildtype, mutant_seq)) if a != b]
    if not diffs:
        return np.zeros(5)

    d_hydro = np.mean([HYDROPHOBICITY.get(mutant_seq[i], 0) - HYDROPHOBICITY.get(wildtype[i], 0) for i in diffs])
    d_charge = np.mean([CHARGE.get(mutant_seq[i], 0) - CHARGE.get(wildtype[i], 0) for i in diffs])
    d_bulky = np.mean([int(mutant_seq[i] in BULKY) - int(wildtype[i] in BULKY) for i in diffs])
    spread = (max(diffs) - min(diffs)) if len(diffs) > 1 else 0

    return np.array([len(diffs), d_hydro, d_charge, d_bulky, spread])


def featurize_assay(wildtype: str, sequences: pd.Series) -> np.ndarray:
    """Build the full baseline feature matrix for an assay.

    28-dim per mutant: 20 (composition) + 3 (global physicochemical)
    + 5 (mutation-specific).
    """
    rows = []
    for seq in sequences:
        comp = amino_acid_composition(seq)
        phys = global_physicochemical_features(seq)
        mut = mutation_features(wildtype, seq)
        rows.append(np.concatenate([comp, phys, mut]))
    return np.vstack(rows)


FEATURE_NAMES = (
    [f"frac_{aa}" for aa in AMINO_ACIDS]
    + ["mean_hydrophobicity", "mean_charge", "frac_bulky"]
    + ["n_mutations", "d_hydrophobicity", "d_charge", "d_bulky", "position_spread"]
)
