"""Summarize paired seeds and verify acceptance without rerunning training."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, mean_absolute_error

from admet_multitask.data import TASKS, leakage_audit


def summarize(output=Path("results")):
    output = Path(output)
    metrics = pd.read_csv(output / "metrics.csv")
    predictions = pd.read_csv(output / "predictions.csv.gz")
    frame = pd.read_csv(output / "molecules.csv.gz", keep_default_na=False)
    audit = json.loads((output / "audit.json").read_text())
    config = json.loads((output / "config.json").read_text())
    assert len(metrics) == 30
    assert not metrics.duplicated(["seed", "method", "task"]).any()
    assert set(metrics.seed) == set(config["seeds"])
    assert np.isfinite(metrics.value).all()
    leakage_audit(frame)
    assert not frame.smiles.duplicated().any()
    for row in metrics.itertuples():
        p = predictions[(predictions.seed == row.seed) & (predictions.method == row.method)
                        & (predictions.task == row.task)]
        assert len(p) == row.test_n and not p.molecule_id.duplicated().any()
        assert (frame.loc[p.molecule_id, "split"] == "test").all()
        np.testing.assert_allclose(frame.loc[p.molecule_id, row.task].astype(float), p.actual)
        assert np.isfinite(p.prediction).all()
        function = average_precision_score if row.task == "cyp2d6" else mean_absolute_error
        np.testing.assert_allclose(function(p.actual, p.prediction), row.value, rtol=1e-7)
    rows, differences = [], []
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, task in zip(axes, TASKS):
        paired = metrics[metrics.task == task].pivot(index="seed", columns="method", values="value")
        st, mt = paired.single_task.to_numpy(), paired.multitask.to_numpy()
        gain = mt - st if task == "cyp2d6" else st - mt
        # Two-sided simultaneous intervals: 1 - alpha / three endpoints.
        confidence = 1 - 0.05 / len(TASKS)
        half = student_t.ppf((1 + confidence) / 2, len(gain) - 1) * gain.std(ddof=1) / np.sqrt(len(gain))
        low, high = float(gain.mean() - half), float(gain.mean() + half)
        effect = float(gain.mean() if task == "cyp2d6" else gain.mean() / st.mean())
        threshold = 0.02 if task == "cyp2d6" else 0.05
        supported = bool(low > 0 and effect >= threshold)
        rows.append({"task": task, "metric": "AP" if task == "cyp2d6" else "MAE",
                     "test_n": int(metrics.loc[metrics.task == task, "test_n"].iloc[0]),
                     "single_task_mean": st.mean(), "single_task_std": st.std(ddof=1),
                     "multitask_mean": mt.mean(), "multitask_std": mt.std(ddof=1),
                     "paired_gain_mean": gain.mean(), "paired_gain_std": gain.std(ddof=1),
                     "simultaneous_ci_low": low, "simultaneous_ci_high": high,
                     "confidence": confidence, "effect": effect,
                     "effect_threshold": threshold, "help_supported": supported})
        differences.extend({"task": task, "seed": int(seed), "gain": float(g)}
                           for seed, g in zip(paired.index, gain))
        for a, b in zip(st, mt):
            axis.plot([0, 1], [a, b], "o-", color="0.65", alpha=0.8, markersize=4)
        axis.errorbar([0, 1], [st.mean(), mt.mean()], yerr=[st.std(ddof=1), mt.std(ddof=1)],
                      fmt="s", capsize=5, color="#185f8c", markersize=7, linewidth=2)
        axis.set_xticks([0, 1], ["Single task", "Multitask"])
        axis.set_title(task)
        axis.set_ylabel("AP (higher is better)" if task == "cyp2d6" else "MAE (lower is better)")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Global scaffold split · paired seeds · mean ± sample SD")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(differences).to_csv(output / "paired_differences.csv", index=False)
    lines = ["| Endpoint | Metric | Test labels | Single task | Multitask | Paired gain [simultaneous CI] | Help supported? |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['metric']} | {r['test_n']} | "
                     f"{r['single_task_mean']:.4f} ± {r['single_task_std']:.4f} | "
                     f"{r['multitask_mean']:.4f} ± {r['multitask_std']:.4f} | "
                     f"{r['paired_gain_mean']:.4f} [{r['simultaneous_ci_low']:.4f}, {r['simultaneous_ci_high']:.4f}] | "
                     f"{'Yes' if r['help_supported'] else 'No'} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    manifest = json.loads((output / "data_manifest.json").read_text())
    acceptance = {"datasets_expected": 3, "datasets_actual": len(manifest),
                  "metric_records_expected": 30, "metric_records_actual": len(metrics),
                  "prediction_records_expected": int(metrics.test_n.sum()),
                  "prediction_records_actual": len(predictions),
                  "leakage_checks_expected": 6, "leakage_checks_actual": len(audit["leakage"]),
                  "overlap_total": sum(audit["leakage"].values()),
                  "minimum_labels_per_task_partition": min(n for counts in audit["labels"].values() for n in counts.values()),
                  "supported_endpoints": int(summary.help_supported.sum()),
                  "general_help_supported": bool(summary.help_supported.all()),
                  "local_experiment_passed": True}
    assert acceptance["datasets_actual"] == 3 and acceptance["overlap_total"] == 0
    (output / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    artifacts = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(output.iterdir()) if p.is_file() and p.name != "artifact_manifest.json"}
    (output / "artifact_manifest.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    print((output / "comparison.md").read_text())
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    summarize(parser.parse_args().output)
