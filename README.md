# Protein Fitness Prediction Benchmark

A small, reproducible benchmark for the core question behind AI-for-biology
tools like [GPT-Rosalind](https://openai.com/index/introducing-gpt-rosalind/),
[ESM](https://github.com/facebookresearch/esm), and AlphaFold-adjacent
models: **given a protein sequence and a mutation, can a model predict how
that mutation affects the protein's fitness** (stability, binding,
catalytic activity) **without running the experiment?**

This matters because Deep Mutational Scanning (DMS) — the gold-standard way
to measure this experimentally — is slow and expensive: you synthesize
thousands of variants and assay each one. A model that ranks variants well
*before* the wet lab means fewer variants need to be tested to find the
useful ones.

## What this repo does

1. Loads DMS assay data in the standard format used by
   [ProteinGym](https://github.com/OATML-Markslab/ProteinGym) (mutant,
   mutated sequence, experimental fitness score).
2. Extracts two tiers of features:
   - **Baseline** (`src/features.py`): amino acid composition,
     physicochemical properties (Kyte-Doolittle hydrophobicity, charge,
     bulkiness), and mutation-specific deltas. No pretrained model, no
     GPU, no internet — this is the floor any real model needs to beat.
   - **Protein language model embeddings** (`src/embeddings.py`, optional):
     ESM-2 embeddings as a stronger baseline, for when you have GPU access.
3. Fits a simple ridge regression per assay with 5-fold cross-validation
   and reports **Spearman rank correlation** between predicted and true
   fitness — the standard metric in this field, since what matters for
   variant selection is getting the *ranking* right, not the exact value.
4. Aggregates results across assays into a single leaderboard-style table,
   mirroring how ProteinGym itself reports results.

## Why it's structured this way

The featurizer, embedding, and evaluation steps are deliberately separate
modules so it's easy to swap in a new predictor — including calling out to
GPT-Rosalind, GPT-5-class models, or any other API — and get an
apples-to-apples comparison against the same held-out assays and the same
metric everyone else in the field uses.

## Quickstart

```bash
pip install -r requirements.txt

# Runs on synthetic data — no download, no internet needed, ~2 seconds.
python run_benchmark.py --demo

# Run the test suite (checks the pipeline actually recovers a known signal):
pytest tests/ -v
```

## Using real data

The full ProteinGym benchmark (~2.7M mutations across 217 assays) is
hosted by the Marks Lab at Harvard and is too large to bundle here. To use
real assays instead of the synthetic demo:

```bash
VERSION="v1.3"
FILENAME="DMS_ProteinGym_substitutions.zip"
curl -o ${FILENAME} https://marks.hms.harvard.edu/proteingym/ProteinGym_${VERSION}/${FILENAME}
unzip ${FILENAME} -d data/
rm ${FILENAME}

python run_benchmark.py --data-dir data/DMS_ProteinGym_substitutions --limit 10
```

(`--limit 10` runs on the first 10 assays only — drop it to run the full
benchmark, but expect it to take a while on CPU.)

## Using the embedding baseline

```bash
pip install torch fair-esm
```

```python
from src.embeddings import embed_sequences
from src.data import make_synthetic_assay

assay = make_synthetic_assay()
X = embed_sequences(assay.df["mutated_sequence"].tolist())
```

Swap `embed_sequences(..., model_name="esm2_t33_650M_UR50D")` for a larger
model if you have GPU headroom — see `src/embeddings.py` for the full list
of ESM-2 checkpoint sizes.

## Results

### Real data (ProteinGym, 5 assays)

The baseline features get a modest but clearly non-random signal on real
Deep Mutational Scanning data — well below what a protein language model
like ESM or EVE achieves (published ProteinGym baselines typically land
in the 0.3–0.6 Spearman range), which is the expected and honest result
for hand-crafted composition/physicochemical features versus a learned
representation. That gap is the interesting part: it's a concrete,
reproducible demonstration of how much signal a pretrained protein
language model adds over simple sequence statistics.

```
assay_name                           n_mutants  spearman
A0A140D2T1_ZIKV_Sourisseau_2019        9576       0.111
A0A192B1T2_9HIV1_Haddox_2018           12577      0.275
A0A1I9GEU1_NEIME_Kennouche_2019        922       -0.066
A0A247D711_LISMN_Stadelmann_2021       1653       0.247
A0A2Z5U3Z0_9INFA_Doud_2016             10715      0.186
MEAN                                   35443      0.150
```

Run it yourself: `python run_benchmark.py --data-dir data/DMS_ProteinGym_substitutions --limit 5`

### Synthetic demo (sanity check, not a real result)

`python run_benchmark.py --demo` runs on a synthetic assay with a known,
designed-in signal — this checks that the pipeline correctly recovers a
signal it should be able to find, not that real protein fitness
prediction is easy. Expect Spearman ≈0.95 here; it is **not** comparable
to the real-data numbers above.

## Project structure

```
src/
  data.py        # assay loading + synthetic data generator
  features.py     # composition / physicochemical baseline features
  embeddings.py   # optional ESM-2 embedding baseline
  evaluate.py     # cross-validated Spearman evaluation harness
tests/
  test_pipeline.py
run_benchmark.py  # CLI entry point
```

## References

- Notin et al., "ProteinGym: Large-Scale Benchmarks for Protein Fitness
  Prediction and Design," NeurIPS 2023.
- Kyte & Doolittle, "A simple method for displaying the hydropathic
  character of a protein," J Mol Biol, 1982.
- Lin et al., "Evolutionary-scale prediction of atomic-level protein
  structure with a language model" (ESM-2), Science, 2023.
