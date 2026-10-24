"""Validate committed evidence against the actual saved test predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error

from admet_multitask.data import TASKS, leakage_audit, train_scaling

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def test_artifacts_present_and_hashes_match():
    manifest = json.loads((RESULTS / "artifact_manifest.json").read_text())
    assert len(manifest) >= 13
    for name, evidence in manifest.items():
        raw = (RESULTS / name).read_bytes()
        assert len(raw) == evidence["bytes"]
        assert hashlib.sha256(raw).hexdigest() == evidence["sha256"]


def test_actual_global_split_and_label_support():
    frame = pd.read_csv(RESULTS / "molecules.csv.gz", keep_default_na=False)
    assert not frame.smiles.duplicated().any()
    assert sum(leakage_audit(frame).values()) == 0
    labels = frame[TASKS].replace("", np.nan).astype(float)
    audit = json.loads((RESULTS / "audit.json").read_text())
    for task in TASKS:
        for split in ["train", "valid", "test"]:
            observed = labels.loc[frame.split == split, task].dropna()
            assert len(observed) == audit["labels"][task][split]
            assert len(observed) >= 30
            if task == "cyp2d6":
                assert set(observed) == {0, 1}
    mean, scale = train_scaling(labels.to_numpy(), frame.split.to_numpy() == "train")
    np.testing.assert_allclose(mean, audit["training_target_means"])
    np.testing.assert_allclose(scale, audit["training_target_scales"])


def test_all_metrics_recomputed_from_predictions():
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    predictions = pd.read_csv(RESULTS / "predictions.csv.gz")
    frame = pd.read_csv(RESULTS / "molecules.csv.gz")
    assert len(metrics) == 30
    assert not metrics.duplicated(["method", "task", "seed"]).any()
    assert set(metrics.seed) == {11, 22, 33, 44, 55}
    assert set(metrics.task) == set(TASKS)
    assert set(metrics.method) == {"single_task", "multitask"}
    for row in metrics.itertuples():
        subset = predictions[(predictions.seed == row.seed) & (predictions.task == row.task)
                             & (predictions.method == row.method)]
        assert len(subset) == row.test_n
        assert not subset.molecule_id.duplicated().any()
        ids = subset.molecule_id
        assert (frame.loc[ids, "split"] == "test").all()
        np.testing.assert_allclose(frame.loc[ids, row.task], subset.actual)
        score = (average_precision_score if row.task == "cyp2d6" else mean_absolute_error)(
            subset.actual, subset.prediction)
        np.testing.assert_allclose(score, row.value, rtol=1e-7)
    assert len(predictions) == metrics.test_n.sum()


def test_summary_and_claims_follow_frozen_rule():
    from scipy.stats import t
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    summary = pd.read_csv(RESULTS / "comparison.csv")
    assert len(summary) == 3
    for row in summary.itertuples():
        pairs = metrics[metrics.task == row.task].pivot(index="seed", columns="method", values="value")
        gain = (pairs.multitask - pairs.single_task) if row.task == "cyp2d6" else (pairs.single_task - pairs.multitask)
        half = t.ppf(1 - .05 / 6, 4) * gain.std(ddof=1) / np.sqrt(5)
        np.testing.assert_allclose([row.paired_gain_mean, row.simultaneous_ci_low, row.simultaneous_ci_high],
                                   [gain.mean(), gain.mean() - half, gain.mean() + half])
        effect = gain.mean() if row.task == "cyp2d6" else gain.mean() / pairs.single_task.mean()
        assert row.help_supported == (gain.mean() - half > 0 and effect >= row.effect_threshold)
