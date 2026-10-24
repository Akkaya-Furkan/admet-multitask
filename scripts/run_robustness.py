"""P24 follow-up: frozen equal-budget search and five paired scaffold splits."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, mean_absolute_error

from admet_multitask.data import TASKS, assign_split, curate, fingerprints, leakage_audit, train_scaling
from admet_multitask.model import fit, observed_loss
from admet_multitask.robustness import nearest_neighbors, paired_bootstrap, select_candidate, tautomer_keys
from run_experiment import CONFIG

PLAN = {"split_seeds": [20261024, 20261025, 20261026, 20261027, 20261028],
        "model_seeds": [11, 22, 33, 44, 55], "search_seed": 101,
        "candidates": [{"learning_rate": .001, "weight_decay": .0001},
                       {"learning_rate": .0003, "weight_decay": .0001},
                       {"learning_rate": .001, "weight_decay": .001}],
        "bootstrap_draws": 20000, "bootstrap_seed": 7301,
        "selection": "minimum mean three-task normalized validation loss; first candidate breaks ties",
        "budget": "3 complete three-task candidate bundles per approach per split; same epochs, patience, batches, architecture and search seed; not equal FLOPs or wall time"}


def validation_losses(model, x, y, split, tasks):
    with torch.no_grad():
        return {t: float(observed_loss(model(torch.from_numpy(x[split == 'valid']), t),
                                      torch.from_numpy(y[split == 'valid', t]), t == 2)) for t in tasks}


def run(raw_dir, output):
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'plan.json').write_text(json.dumps({**PLAN, 'training': CONFIG}, indent=2) + '\n')
    torch.set_num_threads(CONFIG['threads'])
    torch.use_deterministic_algorithms(True)
    frame, curation = curate(raw_dir)
    keys, limited = tautomer_keys(frame.smiles)
    frame['tautomer'] = keys
    frame['tautomer_limited'] = limited
    frame.to_csv(output / 'molecules.csv.gz', index=False)
    x = fingerprints(frame.smiles)
    y = frame[TASKS].to_numpy(dtype=np.float32)
    metrics, predictions, searches, training, audits, neighbors, assignments = [], [], [], [], [], [], []
    for split_seed in PLAN['split_seeds']:
        frame['split'] = assign_split(frame, split_seed)
        split = frame.split.to_numpy()
        leakage = leakage_audit(frame)
        counts = {t: {s: int(frame.loc[frame.split == s, t].notna().sum())
                      for s in ['train', 'valid', 'test']} for t in TASKS}
        assert min(n for v in counts.values() for n in v.values()) >= 30
        for s in ['train', 'valid', 'test']:
            assert set(frame.loc[frame.split == s, 'cyp2d6'].dropna()) == {0, 1}
        tautomer_overlap = {f'{a}_{b}': len(set(frame.loc[frame.split == a, 'tautomer']) &
                                               set(frame.loc[frame.split == b, 'tautomer']))
                            for a, b in combinations(['train', 'valid', 'test'], 2)}
        audits.append({'split_seed': split_seed, 'leakage': leakage, 'labels': counts,
                       'tautomer_overlap': tautomer_overlap, 'tautomer_enumeration_limited': int(sum(limited))})
        assignments.extend({'split_seed': split_seed, 'molecule_id': i, 'split': s}
                           for i, s in enumerate(split))
        nn = nearest_neighbors(frame.smiles, split)
        nn['split_seed'] = split_seed
        neighbors.append(nn)
        means, scales = train_scaling(y, split == 'train')
        normalized = ((y - means) / scales).astype(np.float32)
        # Test labels cannot enter any fit or candidate score, including stopping.
        fit_y = normalized.copy()
        fit_y[split == 'test'] = np.nan
        selected = {}
        for method in ['single_task', 'multitask']:
            groups = [[0], [1], [2]] if method == 'single_task' else [[0, 1, 2]]
            scores = []
            for candidate, params in enumerate(PLAN['candidates']):
                losses = {}
                for tasks in groups:
                    model, info = fit(x, fit_y, split, tasks, PLAN['search_seed'], {**CONFIG, **params})
                    losses.update(validation_losses(model, x, fit_y, split, tasks))
                    training.append({'phase': 'search', 'split_seed': split_seed, 'method': method,
                                     'candidate': candidate, 'seed': PLAN['search_seed'], 'tasks': tasks, **info})
                scores.append([losses[t] for t in range(3)])
            winner = select_candidate(scores)
            selected[method] = winner
            searches.extend({'split_seed': split_seed, 'method': method, 'candidate': c,
                             'validation_losses': score, 'score': float(np.mean(score)), 'selected': c == winner}
                            for c, score in enumerate(scores))
        # Both approaches' choices are frozen before reading test labels for scoring.
        (output / f'selection_{split_seed}.json').write_text(json.dumps(searches[-6:], indent=2) + '\n')
        for seed in PLAN['model_seeds']:
            for method in ['single_task', 'multitask']:
                candidate = selected[method]
                groups = [[0], [1], [2]] if method == 'single_task' else [[0, 1, 2]]
                for tasks in groups:
                    model, info = fit(x, fit_y, split, tasks, seed, {**CONFIG, **PLAN['candidates'][candidate]})
                    training.append({'phase': 'evaluation', 'split_seed': split_seed, 'method': method,
                                     'candidate': candidate, 'seed': seed, 'tasks': tasks, **info})
                    for t in tasks:
                        ids = np.flatnonzero((split == 'test') & np.isfinite(y[:, t]))
                        with torch.no_grad():
                            p = model(torch.from_numpy(x[ids]), t)
                            p = torch.sigmoid(p).numpy() if t == 2 else p.numpy() * scales[t] + means[t]
                        value = (average_precision_score if t == 2 else mean_absolute_error)(y[ids, t], p)
                        metrics.append({'split_seed': split_seed, 'seed': seed, 'method': method,
                                        'task': TASKS[t], 'candidate': candidate, 'test_n': len(ids), 'value': float(value)})
                        predictions.extend({'split_seed': split_seed, 'seed': seed, 'method': method,
                                            'task': TASKS[t], 'molecule_id': int(i), 'actual': float(a), 'prediction': float(b)}
                                           for i, a, b in zip(ids, y[ids, t], p))
        print(f'split {split_seed}: {len(metrics)}/150 metrics; elapsed {time.perf_counter()-start:.2f}s', flush=True)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(output / 'predictions.csv.gz', index=False)
    pd.DataFrame(assignments).to_csv(output / 'splits.csv.gz', index=False)
    (output / 'search.json').write_text(json.dumps(searches, indent=2) + '\n')
    (output / 'training.json').write_text(json.dumps(training, indent=2) + '\n')
    (output / 'audit.json').write_text(json.dumps({'curation': curation, 'splits': audits}, indent=2) + '\n')
    nn = pd.concat(neighbors, ignore_index=True)
    nn.to_csv(output / 'nearest_neighbors.csv.gz', index=False)
    rows, differences = [], []
    for task in TASKS:
        pairs = metric_frame[metric_frame.task == task].pivot(index=['split_seed', 'seed'], columns='method', values='value')
        gain = pairs.multitask - pairs.single_task if task == 'cyp2d6' else pairs.single_task - pairs.multitask
        result = paired_bootstrap(gain.unstack().to_numpy(), PLAN['bootstrap_draws'], PLAN['bootstrap_seed'])
        effect = result['gain'] if task == 'cyp2d6' else result['gain'] / pairs.single_task.mean()
        threshold = .02 if task == 'cyp2d6' else .05
        rows.append({'task': task, 'single_task_mean': pairs.single_task.mean(),
                     'multitask_mean': pairs.multitask.mean(), **result, 'effect': effect,
                     'help_supported': bool(result['simultaneous_ci_low'] > 0 and effect >= threshold)})
        differences.extend({'task': task, 'split_seed': s, 'seed': r, 'gain': float(g)} for (s, r), g in gain.items())
    pd.DataFrame(rows).to_csv(output / 'comparison.csv', index=False)
    pd.DataFrame(differences).to_csv(output / 'paired_differences.csv', index=False)
    distributions = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for seed, subset in nn.groupby('split_seed'):
        values = np.sort(subset.tanimoto)
        axes[1].plot(values, np.arange(1, len(values)+1)/len(values), label=str(seed))
        for task in ['union', *TASKS]:
            v = subset if task == 'union' else subset[frame.loc[subset.molecule_id, task].notna().to_numpy()]
            distributions.append({'split_seed': seed, 'task': task, 'n': len(v),
                                  **{f'q{q}': float(v.tanimoto.quantile(q/100)) for q in [0, 25, 50, 75, 90, 95, 100]},
                                  'ge_0.8': int((v.tanimoto >= .8).sum()), 'ge_0.9': int((v.tanimoto >= .9).sum()),
                                  'equal_1': int((v.tanimoto == 1).sum())})
    pd.DataFrame(distributions).to_csv(output / 'similarity_summary.csv', index=False)
    for i, row in enumerate(rows):
        axes[0].errorbar(i, row['gain'], yerr=[[row['gain']-row['ci95_low']], [row['ci95_high']-row['gain']]], fmt='o', capsize=5)
    axes[0].axhline(0, color='gray', linestyle='--')
    axes[0].set_xticks(range(3), TASKS)
    axes[0].set_ylabel('Paired gain, hierarchical bootstrap 95% CI')
    axes[1].set(xlabel='Nearest training Morgan Tanimoto', ylabel='Test molecule empirical CDF')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / 'robustness.png', dpi=180)
    plt.close(fig)
    metadata = {'elapsed_seconds': time.perf_counter()-start, 'python': platform.python_version(),
                'versions': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'pandas', 'rdkit', 'scikit-learn']},
                'metric_records_expected': 150, 'metric_records_actual': len(metrics),
                'prediction_records_expected': int(metric_frame.test_n.sum()), 'prediction_records_actual': len(predictions),
                'search_bundles_expected': 30, 'search_bundles_actual': len(searches),
                'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*Path('src').rglob('*.py'), Path(__file__)]},
                'data_manifest': json.loads((Path(raw_dir)/'manifest.json').read_text())}
    assert len(metrics) == 150 and len(searches) == 30 and len(predictions) == metric_frame.test_n.sum()
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    manifest = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(output.iterdir()) if p.is_file() and p.name != 'artifact_manifest.json'}
    (output / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--output', default='results/robustness')
    args = parser.parse_args()
    run(args.raw_dir, args.output)
