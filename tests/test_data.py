import numpy as np
import pandas as pd
import pytest

from admet_multitask.data import (canonical_parent, scaffold, assign_split,
                                 leakage_audit, fingerprints, train_scaling, curate, TASKS)


def test_parent_identity_and_invalid():
    assert canonical_parent("CCO.[Na+]") == canonical_parent("OCC")
    assert canonical_parent("C[C@H](O)F") == canonical_parent("C[C@@H](O)F")
    assert canonical_parent("invalid") is None
    assert canonical_parent("") is None
    assert canonical_parent("CC(=O)[c-]1c(=O)cc(C)oc1=O.[Na+]") is None


def test_scaffolds_and_acyclic_group():
    assert scaffold("Cc1ccccc1") == scaffold("Oc1ccccc1")
    assert scaffold("CCO") == scaffold("CCCC") == ""


def test_global_split_is_deterministic_label_blind_and_disjoint():
    frame = pd.DataFrame({"smiles": [f"identity{i}" for i in range(100)],
                          "scaffold": [f"scaffold{i // 3}" for i in range(100)],
                          "label": np.arange(100)})
    frame["split"] = assign_split(frame)
    assert set(frame.split) == {"train", "valid", "test"}
    assert (assign_split(frame.assign(label=-999)) == frame.split).all()
    assert sum(leakage_audit(frame).values()) == 0
    assert frame.groupby("scaffold").split.nunique().max() == 1


@pytest.mark.parametrize("column", ["smiles", "scaffold"])
def test_leakage_is_rejected(column):
    frame = pd.DataFrame({"smiles": ["a", "b", "c"], "scaffold": ["x", "y", "z"],
                          "split": ["train", "valid", "test"]})
    frame.loc[1, column] = frame.loc[0, column]
    with pytest.raises(ValueError, match="Leakage"):
        leakage_audit(frame)


def test_scaling_ignores_validation_test_and_missing_labels():
    y = np.array([[1, 10, 0], [3, np.nan, 1], [999, -999, 1]], dtype=float)
    mean, std = train_scaling(y, np.array([True, True, False]))
    np.testing.assert_allclose(mean, [2, 10, 0])
    np.testing.assert_allclose(std, [1, 1e-8, 1])


def test_fingerprints():
    result = fingerprints(["CCO", "OCC", "c1ccccc1"], size=128)
    assert result.shape == (3, 128)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[0], result[1])
    assert set(np.unique(result)) == {0, 1}


def test_duplicate_aggregation_conflicts_and_union(tmp_path):
    for task in TASKS:
        values = [0, 1, 1] if task == "cyp2d6" else [1, 3, 4]
        pd.DataFrame({"Drug": ["CCO", "OCC", "c1ccccc1"], "Y": values}).to_csv(
            tmp_path / f"{task}.tab", sep="\t", index=False)
    frame, audit = curate(tmp_path)
    assert len(frame) == 2
    assert frame.loc[frame.smiles == "CCO", "solubility"].item() == 2
    assert np.isnan(frame.loc[frame.smiles == "CCO", "cyp2d6"].item())
    assert audit["cyp2d6"]["conflicting_molecules_dropped"] == 1
    assert audit["solubility"]["duplicate_rows_aggregated"] == 1
