"""
Protein language model embeddings, used as a stronger baseline than the
hand-crafted features in features.py.

This is intentionally optional and isolated in its own module: it needs
`torch` + `fair-esm` installed and either a GPU or patience, plus internet
access the first time (to download pretrained weights from Meta). None of
that is available in a locked-down sandbox, so evaluate.py falls back to
the hand-crafted baseline if this import fails -- but on your own machine
or Colab (with `pip install fair-esm torch`), this is the model you'd
actually want to compare against, and against GPT-Rosalind / GPT-5-class
models via the API.
"""
from __future__ import annotations

import numpy as np

_MODEL = None
_ALPHABET = None
_BATCH_CONVERTER = None


def _lazy_load(model_name: str = "esm2_t12_35M_UR50D"):
    """Load an ESM-2 model on first use. Small variant by default (35M
    params) so it's runnable without a big GPU; swap for esm2_t33_650M_UR50D
    or larger if you have the compute."""
    global _MODEL, _ALPHABET, _BATCH_CONVERTER
    if _MODEL is not None:
        return

    try:
        import torch  # noqa: F401
        import esm
    except ImportError as e:
        raise ImportError(
            "Embedding baseline requires `torch` and `fair-esm`. "
            "Install with: pip install torch fair-esm"
        ) from e

    _MODEL, alphabet = getattr(esm.pretrained, model_name)()
    _MODEL.eval()
    _ALPHABET = alphabet
    _BATCH_CONVERTER = alphabet.get_batch_converter()


def embed_sequences(sequences: list[str], model_name: str = "esm2_t12_35M_UR50D", batch_size: int = 8) -> np.ndarray:
    """Return mean-pooled per-residue embeddings, one row per sequence."""
    import torch  # local import: only required if this function is called

    _lazy_load(model_name)
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = [(str(j), seq) for j, seq in enumerate(sequences[i : i + batch_size])]
            _, _, tokens = _BATCH_CONVERTER(batch)
            out = _MODEL(tokens, repr_layers=[_MODEL.num_layers])
            reprs = out["representations"][_MODEL.num_layers]

            for k, (_, seq) in enumerate(batch):
                # mean-pool over real residues, excluding BOS/EOS tokens
                emb = reprs[k, 1 : len(seq) + 1].mean(0)
                all_embeddings.append(emb.numpy())

    return np.vstack(all_embeddings)
