"""Follow-up protocol and recomputation of committed evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score, mean_absolute_error

from admet_multitask.data import TASKS, leakage_audit
from admet_multitask.robustness import nearest_neighbors, paired_bootstrap, select_candidate, tautomer_keys

RESULTS = Path(__file__).resolve().parents[1] / 'results' / 'robustness'


def test_selection_uses_equal_task_weight_and_stable_ties():
    assert select_candidate([[1, 2, 3], [2, 2, 2], [1, 1, 1]]) == 2
    assert select_candidate([[1, 2, 3], [2, 2, 2]]) == 0
    with pytest.raises(ValueError):
        select_candidate([[1, np.nan, 2]])


def test_bootstrap_pairs_and_split_clusters():
    constant = paired_bootstrap(np.full((5, 5), .125))
    assert all(v == .125 for v in constant.values())
    # Seeds within one split do not create independent evidence about splits.
    a = paired_bootstrap(np.repeat(np.array([-2, -1, 0, 1, 2])[:, None], 5, axis=1))
    assert a['gain'] == 0 and a['ci95_low'] < 0 < a['ci95_high']
    assert a['simultaneous_ci_low'] <= a['ci95_low']
    assert a == paired_bootstrap(np.repeat(np.array([-2, -1, 0, 1, 2])[:, None], 5, axis=1))


def test_chemical_audit_live_path():
    smiles = ['CCO', 'c1ccccc1', 'CCO']
    nn = nearest_neighbors(smiles, ['train', 'train', 'test'])
    assert nn.iloc[0].neighbor_id == 0 and nn.iloc[0].tanimoto == 1
    keys, limited = tautomer_keys(['CC(=O)C', 'CC(O)=C'])
    assert keys[0] == keys[1] and not any(limited)


def test_followup_artifacts_counts_and_search_budget():
    manifest = json.loads((RESULTS / 'artifact_manifest.json').read_text())
    for name, evidence in manifest.items():
        raw = (RESULTS / name).read_bytes()
        assert len(raw) == evidence['bytes']
        assert hashlib.sha256(raw).hexdigest() == evidence['sha256']
    meta = json.loads((RESULTS / 'metadata.json').read_text())
    assert meta['metric_records_actual'] == meta['metric_records_expected'] == 150
    assert meta['search_bundles_actual'] == meta['search_bundles_expected'] == 30
    search = pd.DataFrame(json.loads((RESULTS / 'search.json').read_text()))
    plan = json.loads((RESULTS / 'plan.json').read_text())
    assert plan['search_seed'] not in plan['model_seeds']
    for (_, _), group in search.groupby(['split_seed', 'method']):
        group = group.sort_values('candidate')
        assert list(group.candidate) == [0, 1, 2]
        winner = select_candidate(group.validation_losses.tolist())
        assert group.selected.sum() == 1
        assert group.loc[group.selected, 'candidate'].item() == winner
    training = json.loads((RESULTS / 'training.json').read_text())
    assert len(training) == 160  # 60 search fits + 100 final fits
    for row in training:
        if row['phase'] == 'search':
            assert row['seed'] == plan['search_seed']
        else:
            assert row['seed'] in plan['model_seeds']
    for s in plan['split_seeds']:
        for method in ['single_task', 'multitask']:
            # Same candidate count and per-task fit exposure, despite 3 ST networks.
            for task in range(3):
                fits = [r for r in training if r['phase'] == 'search' and r['split_seed'] == s
                        and r['method'] == method and task in r['tasks']]
                assert len(fits) == 3


def test_followup_metrics_splits_and_intervals_recomputed():
    metrics = pd.read_csv(RESULTS / 'metrics.csv')
    predictions = pd.read_csv(RESULTS / 'predictions.csv.gz')
    frame = pd.read_csv(RESULTS / 'molecules.csv.gz', keep_default_na=False)
    splits = pd.read_csv(RESULTS / 'splits.csv.gz')
    plan = json.loads((RESULTS / 'plan.json').read_text())
    assert len(metrics) == 150 and not metrics.duplicated(['split_seed', 'seed', 'method', 'task']).any()
    assert len(predictions) == metrics.test_n.sum()
    assert len(splits) == len(frame) * 5
    signatures = []
    for seed, group in splits.groupby('split_seed'):
        group = group.sort_values('molecule_id')
        assert list(group.molecule_id) == list(range(len(frame)))
        current = frame.assign(split=group.split.to_numpy())
        assert sum(leakage_audit(current).values()) == 0
        signatures.append(tuple(group.split))
        for task in TASKS:
            for part in ['train', 'valid', 'test']:
                labels = current.loc[current.split == part, task].replace('', np.nan).astype(float).dropna()
                assert len(labels) >= 30
                if task == 'cyp2d6':
                    assert set(labels) == {0, 1}
    assert len(set(signatures)) == 5
    assignments = splits.set_index(['split_seed', 'molecule_id']).split
    for row in metrics.itertuples():
        p = predictions[(predictions.split_seed == row.split_seed) & (predictions.seed == row.seed)
                        & (predictions.method == row.method) & (predictions.task == row.task)]
        assert len(p) == row.test_n and not p.molecule_id.duplicated().any()
        assert np.isfinite(p.prediction).all()
        assert all(assignments.loc[(row.split_seed, i)] == 'test' for i in p.molecule_id)
        np.testing.assert_allclose(frame.loc[p.molecule_id, row.task].astype(float), p.actual)
        metric = average_precision_score if row.task == 'cyp2d6' else mean_absolute_error
        np.testing.assert_allclose(metric(p.actual, p.prediction), row.value, rtol=1e-7)
    summary = pd.read_csv(RESULTS / 'comparison.csv')
    for row in summary.itertuples():
        pairs = metrics[metrics.task == row.task].pivot(index=['split_seed', 'seed'], columns='method', values='value')
        gain = pairs.multitask - pairs.single_task if row.task == 'cyp2d6' else pairs.single_task - pairs.multitask
        recomputed = paired_bootstrap(gain.unstack().to_numpy(), plan['bootstrap_draws'], plan['bootstrap_seed'])
        for key, value in recomputed.items():
            np.testing.assert_allclose(getattr(row, key), value, atol=1e-12)
        threshold = .02 if row.task == 'cyp2d6' else .05
        effect = gain.mean() if row.task == 'cyp2d6' else gain.mean()/pairs.single_task.mean()
        assert row.help_supported == (row.simultaneous_ci_low > 0 and effect >= threshold)


def test_saved_similarity_neighbors_and_distribution():
    frame = pd.read_csv(RESULTS / 'molecules.csv.gz')
    splits = pd.read_csv(RESULTS / 'splits.csv.gz')
    nn = pd.read_csv(RESULTS / 'nearest_neighbors.csv.gz')
    summary = pd.read_csv(RESULTS / 'similarity_summary.csv')
    assert len(nn) == (splits.split == 'test').sum()
    assert not nn.duplicated(['split_seed', 'molecule_id']).any()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    # Deterministic sample checks saved neighbors against every training molecule.
    for seed, part in nn.groupby('split_seed'):
        partition = splits[splits.split_seed == seed].set_index('molecule_id').split
        assert (partition.loc[part.molecule_id] == 'test').all()
        assert (partition.loc[part.neighbor_id] == 'train').all()
        train_ids = partition.index[partition == 'train']
        fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in frame.loc[train_ids, 'smiles']]
        for row in part.iloc[::500].itertuples():
            fp = generator.GetFingerprint(Chem.MolFromSmiles(frame.loc[row.molecule_id, 'smiles']))
            scores = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps))
            np.testing.assert_allclose(scores.max(), row.tanimoto)
            assert train_ids[int(scores.argmax())] == row.neighbor_id
    for row in summary.itertuples():
        part = nn[nn.split_seed == row.split_seed]
        if row.task != 'union':
            part = part[frame.loc[part.molecule_id, row.task].notna().to_numpy()]
        assert row.n == len(part)
        np.testing.assert_allclose(row.q50, part.tanimoto.median())
        saved = summary[(summary.split_seed == row.split_seed) & (summary.task == row.task)].iloc[0]
        assert saved['ge_0.8'] == (part.tanimoto >= .8).sum()
        assert saved['ge_0.9'] == (part.tanimoto >= .9).sum()
        assert saved['equal_1'] == (part.tanimoto == 1).sum()
        for q in [0, 25, 50, 75, 90, 95, 100]:
            np.testing.assert_allclose(saved[f'q{q}'], part.tanimoto.quantile(q/100))
