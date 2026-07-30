"""
Loading utilities for Deep Mutational Scanning (DMS) protein fitness assays.

A DMS assay measures, for thousands of point mutants of one protein, some
experimental readout of fitness (binding affinity, fluorescence, stability,
viral replication, etc). The standard format used by benchmarks like
ProteinGym (https://github.com/OATML-Markslab/ProteinGym) is a CSV with:

    mutant            e.g. "A123P" or "A123P:D45N" for multi-site mutants
    mutated_sequence  full amino-acid sequence of the mutant
    DMS_score         the experimental fitness readout (higher = more fit)

This module reads that format, and also provides a synthetic generator so
the rest of the pipeline can be developed and tested without needing the
real (multi-GB) ProteinGym download.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class Assay:
    """A single DMS assay: one wild-type protein plus its scored mutants."""
    name: str
    wildtype_sequence: str
    df: pd.DataFrame  # columns: mutant, mutated_sequence, DMS_score


def _infer_wildtype(mutant: str, mutated_sequence: str) -> str:
    """Reconstruct the wildtype sequence by reverting the substitutions
    encoded in a `mutant` string (e.g. "A123P" or "A123P:D45N") against one
    mutated sequence. This is how real ProteinGym CSVs work: they give you
    the mutant sequence plus the mutation code, not the wildtype directly."""
    seq = list(mutated_sequence)
    for part in mutant.split(":"):
        m = re.match(r"^([A-Za-z])(\d+)([A-Za-z])$", part.strip())
        if not m:
            raise ValueError(f"Could not parse mutation code: {part!r}")
        wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
        idx = pos - 1
        if idx >= len(seq):
            raise ValueError(f"Mutation position {pos} out of range for sequence of length {len(seq)}")
        if seq[idx] != mut_aa:
            raise ValueError(
                f"Mismatch at position {pos}: expected mutant residue {mut_aa!r}, "
                f"found {seq[idx]!r} in mutated_sequence"
            )
        seq[idx] = wt_aa
    return "".join(seq)


def load_assay_csv(path: str, name: str | None = None) -> Assay:
    """Load a single ProteinGym-format assay CSV from disk.

    Expects columns: mutant, mutated_sequence, DMS_score.
    The wildtype sequence is determined, in order of preference: (1) a
    `wildtype_sequence` column if present, (2) a companion `<name>_wt.txt`
    file next to the CSV, (3) reconstructed automatically from the first
    row's `mutant` code and `mutated_sequence` -- this is how real
    ProteinGym substitution CSVs work, since they don't ship the wildtype
    sequence as a separate column.
    """
    df = pd.read_csv(path)
    required = {"mutant", "mutated_sequence", "DMS_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    if "wildtype_sequence" in df.columns:
        wt = df["wildtype_sequence"].iloc[0]
    else:
        wt_path = path.rsplit(".", 1)[0] + "_wt.txt"
        try:
            with open(wt_path) as f:
                wt = f.read().strip()
        except FileNotFoundError:
            wt = _infer_wildtype(df["mutant"].iloc[0], df["mutated_sequence"].iloc[0])

    return Assay(name=name or path, wildtype_sequence=wt, df=df)


def make_synthetic_assay(
    n_mutants: int = 500,
    seq_len: int = 80,
    seed: int = 0,
    name: str = "synthetic_demo",
) -> Assay:
    """Generate a toy DMS assay for pipeline testing / CI.

    The synthetic "fitness function" rewards hydrophobic residues in a few
    designated "core" positions and penalizes mutations at a designated
    "catalytic" position -- loosely mimicking how real stability/activity
    assays are structured (a handful of positions dominate the signal, the
    rest is closer to noise). This lets unit tests check that a model can
    recover a known signal, not just that the code runs.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    wildtype = "".join(rng.choice(AMINO_ACIDS) for _ in range(seq_len))
    hydrophobicity = {
        "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
        "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
        "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
        "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
    }
    catalytic_position = rng.randrange(seq_len)

    rows = []
    seen = set()
    while len(rows) < n_mutants:
        pos = rng.randrange(seq_len)
        new_aa = rng.choice(AMINO_ACIDS)
        if new_aa == wildtype[pos] or (pos, new_aa) in seen:
            continue
        seen.add((pos, new_aa))

        mutated = wildtype[:pos] + new_aa + wildtype[pos + 1:]

        d_hydro = hydrophobicity[new_aa] - hydrophobicity[wildtype[pos]]
        score = 0.5 * d_hydro
        if pos == catalytic_position:
            score -= 4.0
        score += np_rng.normal(0, 0.5)

        rows.append(
            {
                "mutant": f"{wildtype[pos]}{pos + 1}{new_aa}",
                "mutated_sequence": mutated,
                "DMS_score": score,
            }
        )

    df = pd.DataFrame(rows)
    return Assay(name=name, wildtype_sequence=wildtype, df=df)
