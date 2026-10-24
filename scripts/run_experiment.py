"""Reproduce the frozen full experiment and persist auditable test outputs."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, mean_absolute_error

from admet_multitask.data import TASKS, curate, assign_split, leakage_audit, fingerprints, train_scaling
from admet_multitask.model import fit

CONFIG = {"seeds": [11, 22, 33, 44, 55], "split_seed": 20261024,
          "fingerprint_bits": 1024, "radius": 2, "hidden": [128, 64],
          "learning_rate": 0.001, "weight_decay": 0.0001,
          "batch_size": 256, "max_epochs": 60, "patience": 10, "threads": 2}


def run(raw_dir, output, config):
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    frame, audit = curate(raw_dir)
    frame["split"] = assign_split(frame, config["split_seed"])
    audit["leakage"] = leakage_audit(frame)
    audit["union_molecules"] = len(frame)
    audit["multi_label_molecules"] = int((frame[TASKS].notna().sum(axis=1) > 1).sum())
    audit["split_molecules"] = frame.groupby("split").size().to_dict()
    audit["split_scaffolds"] = frame.groupby("split").scaffold.nunique().to_dict()
    audit["empty_scaffold_molecules"] = int((frame.scaffold == "").sum())
    audit["labels"] = {}
    for task in TASKS:
        audit["labels"][task] = {}
        for split in ["train", "valid", "test"]:
            values = frame.loc[frame.split == split, task].dropna()
            assert len(values) >= 30
            audit["labels"][task][split] = len(values)
            if task == "cyp2d6":
                assert set(values) == {0, 1}
                audit.setdefault("classification_positives", {})[split] = int(values.sum())
    frame.to_csv(output / "molecules.csv.gz", index=False)
    x = fingerprints(frame.smiles, config["fingerprint_bits"])
    y = frame[TASKS].to_numpy(dtype=np.float32)
    split = frame.split.to_numpy()
    means, scales = train_scaling(y, split == "train")
    normalized = ((y - means) / scales).astype(np.float32)
    audit["training_target_means"] = means.tolist()
    audit["training_target_scales"] = scales.tolist()
    records, predictions, training = [], [], []
    test_ids = np.flatnonzero(split == "test")
    for seed in config["seeds"]:
        for method in ["single_task", "multitask"]:
            task_groups = [[0], [1], [2]] if method == "single_task" else [[0, 1, 2]]
            for tasks in task_groups:
                model, info = fit(x, normalized, split, tasks, seed, config)
                training.append({"seed": seed, "method": method, "tasks": tasks, **info})
                for t in tasks:
                    ids = test_ids[np.isfinite(y[test_ids, t])]
                    with torch.no_grad():
                        pred = model(torch.from_numpy(x[ids]), t)
                        pred = (torch.sigmoid(pred).numpy() if t == 2
                                else pred.numpy() * scales[t] + means[t])
                    metric = average_precision_score(y[ids, t], pred) if t == 2 else mean_absolute_error(y[ids, t], pred)
                    assert np.isfinite(metric)
                    records.append({"seed": seed, "method": method, "task": TASKS[t],
                                    "metric": "AP" if t == 2 else "MAE", "value": float(metric), "test_n": len(ids)})
                    predictions.extend({"seed": seed, "method": method, "task": TASKS[t],
                                        "molecule_id": int(i), "actual": float(a), "prediction": float(p)}
                                       for i, a, p in zip(ids, y[ids, t], pred))
                    print(records[-1], flush=True)
    pd.DataFrame(records).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "predictions.csv.gz", index=False)
    (output / "training.json").write_text(json.dumps(training, indent=2) + "\n")
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy(Path(raw_dir) / "manifest.json", output / "data_manifest.json")
    metadata = {"elapsed_seconds": time.perf_counter() - start, "python": platform.python_version(),
                "platform": platform.platform(),
                "versions": {p: importlib.metadata.version(p) for p in
                             ["torch", "numpy", "pandas", "rdkit", "scipy", "scikit-learn", "matplotlib"]},
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [*Path("src").rglob("*.py"), Path(__file__)]}}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Completed {len(records)} metric records in {metadata['elapsed_seconds']:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    run(args.raw_dir, args.output, CONFIG)
