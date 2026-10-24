"""Parent standardization, task union and global label-blind scaffold split."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

TASKS = ["solubility", "caco2", "cyp2d6"]
RDLogger.DisableLog("rdApp.*")


def canonical_parent(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        # Ignore stereochemistry to keep stereoisomers in the same identity group.
        value = Chem.MolToSmiles(mol, isomericSmiles=False)
        return value if value and mol.GetNumAtoms() and Chem.MolFromSmiles(value) is not None else None
    except (ValueError, RuntimeError):
        return None


def scaffold(smiles):
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)


def curate(raw_dir):
    tables, audit = [], {}
    for task in TASKS:
        raw = pd.read_csv(Path(raw_dir) / f"{task}.tab", sep="\t")
        frame = pd.DataFrame({"smiles": raw.Drug.map(canonical_parent),
                              task: pd.to_numeric(raw.Y, errors="coerce")})
        valid = frame.dropna().copy()
        valid = valid[np.isfinite(valid[task])]
        conflicts = []
        if task == "cyp2d6":
            assert set(valid[task].unique()) <= {0, 1}
            conflicts = valid.groupby("smiles")[task].nunique().loc[lambda s: s > 1].index.tolist()
            valid = valid[~valid.smiles.isin(conflicts)]
        table = valid.groupby("smiles", as_index=False)[task].mean()
        audit[task] = {"raw_rows": len(raw), "invalid_or_nonfinite": len(frame) - len(frame.dropna()[np.isfinite(frame.dropna()[task])]),
                       "conflicting_molecules_dropped": len(conflicts),
                       "valid_rows_after_conflicts": len(valid), "unique_labels": len(table),
                       "duplicate_rows_aggregated": len(valid) - len(table)}
        tables.append(table.set_index("smiles"))
    union = pd.concat(tables, axis=1).sort_index().reset_index()
    union["scaffold"] = union.smiles.map(scaffold)
    return union, audit


def assign_split(frame, seed=20261024):
    """Whole groups, largest first, random tie-breaking; no labels are read."""
    groups = list(frame.groupby("scaffold", sort=True).groups.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    target = np.array([0.70, 0.15, 0.15]) * len(frame)
    sizes = np.zeros(3, dtype=int)
    names = np.array(["train", "valid", "test"])
    result = pd.Series(index=frame.index, dtype=object)
    for indices in groups:
        # Fill train then valid then test to target; oversize groups go to train.
        fitting = np.flatnonzero(sizes + len(indices) <= target)
        partition = int(fitting[0]) if len(fitting) else int(np.argmax(target - sizes))
        result.loc[indices] = names[partition]
        sizes[partition] += len(indices)
    return result


def leakage_audit(frame):
    checks = {}
    for left, right in combinations(["train", "valid", "test"], 2):
        a, b = frame[frame.split == left], frame[frame.split == right]
        for key in ["smiles", "scaffold"]:
            overlap = len(set(a[key]) & set(b[key]))
            checks[f"{left}_{right}_{key}_overlap"] = overlap
            if overlap:
                raise ValueError(f"Leakage: {left}/{right} {key}: {overlap}")
    return checks


def fingerprints(smiles, size=1024):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=size)
    values = np.zeros((len(smiles), size), dtype=np.float32)
    for i, s in enumerate(smiles):
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(Chem.MolFromSmiles(s)), values[i])
    return values


def train_scaling(y, train_mask):
    means, scales = np.zeros(y.shape[1]), np.ones(y.shape[1])
    for t in range(min(2, y.shape[1])):
        observed = y[train_mask, t]
        means[t] = np.nanmean(observed)
        scales[t] = max(float(np.nanstd(observed)), 1e-8)
    return means, scales
