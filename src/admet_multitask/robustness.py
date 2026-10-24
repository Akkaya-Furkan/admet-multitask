"""Validation-only bundle selection, paired cluster bootstrap and chemical audit."""
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize


def select_candidate(scores):
    """Each candidate is a complete three-task approach; smaller is better."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite candidate-by-three-task validation losses")
    return int(np.argmin(values.mean(axis=1)))


def paired_bootstrap(gains, draws=20000, seed=7301):
    """Resample split clusters, then paired model seeds within each cluster.

    Never treat the 25 overlapping-split runs as 25 independent datasets.
    Percentile intervals describe this split algorithm, not external populations.
    """
    values = np.asarray(gains, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("Need a finite split-by-seed matrix with at least 2 of each")
    rng = np.random.default_rng(seed)
    n, k = values.shape
    splits = rng.integers(n, size=(draws, n, 1))
    seeds = rng.integers(k, size=(draws, n, k))
    means = values[splits, seeds].mean(axis=(1, 2))
    return {"gain": float(values.mean()),
            "ci95_low": float(np.quantile(means, .025)),
            "ci95_high": float(np.quantile(means, .975)),
            "simultaneous_ci_low": float(np.quantile(means, .05 / 6)),
            "simultaneous_ci_high": float(np.quantile(means, 1 - .05 / 6))}


def tautomer_keys(smiles):
    enumerator = rdMolStandardize.TautomerEnumerator()
    keys, limited = [], []
    for s in smiles:
        result = enumerator.Enumerate(Chem.MolFromSmiles(s))
        limited.append(result.status != rdMolStandardize.TautomerEnumeratorStatus.Completed)
        keys.append(Chem.MolToSmiles(enumerator.PickCanonical(result), isomericSmiles=False))
    return keys, limited


def nearest_neighbors(smiles, split, size=1024):
    """Exact bit-vector Tanimoto to all training molecules; stable first tie."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=size)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    train = np.flatnonzero(np.asarray(split) == "train")
    test = np.flatnonzero(np.asarray(split) == "test")
    if not len(train) or not len(test):
        raise ValueError("Need train and test molecules")
    refs = [fps[i] for i in train]
    rows = []
    for i in test:
        similarity = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], refs))
        j = int(np.argmax(similarity))
        rows.append({"molecule_id": int(i), "neighbor_id": int(train[j]),
                     "tanimoto": float(similarity[j])})
    return pd.DataFrame(rows)
