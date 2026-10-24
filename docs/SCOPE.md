# Scope and acceptance criteria

## Question
Does sharing a molecular representation help predict aqueous solubility,
Caco-2 permeability and CYP2D6 inhibition, relative to independent networks?

## One-day experiment (2026-10-24)
Use the complete public TDC Solubility_AqSolDB, Caco2_Wang and CYP2D6_Veith
files from Harvard Dataverse. Use standardized parent canonical SMILES,
merge repeated measurements (mean for regression; discard conflicting binary
labels), and unite molecules across tasks before splitting. Generate Morgan
fingerprints and compare independent single-task MLPs with one shared-trunk,
three-head MLP. Hidden architecture, optimizer, maximum epochs and validation
selection rule are identical. Regression targets are standardized on training
labels only; missing labels are masked; each task contributes equal loss weight.

Freeze one label-blind, global Bemis–Murcko scaffold split, targeting 70/15/15.
The empty scaffold is one group (acyclic molecules stay together). Repeat
initialization/minibatch seeds 11, 22, 33, 44, 55 on this same split. Report
MAE for regression and average precision (AP) for classification, mean ± sample
standard deviation, all paired differences, and paired t intervals across seeds.
The intervals measure training randomness on a fixed split, not population or
split uncertainty. No test-set tuning or selection is allowed.

## Numeric acceptance criteria (defined before training)
- Download 3/3 datasets with URL, byte count and SHA-256 provenance.
- Each task has at least 30 observed train, validation and test labels; the
  classification task has both classes in every partition.
- Zero canonical-molecule and scaffold intersections between every pair of
  partitions, including across tasks. Persist identities, split and audit.
- Complete 5 seeds × 2 approaches × 3 tasks = 30 test metric records and
  save test predictions. Every metric must be finite.
- Produce one machine-readable comparison table and one comparison plot,
  plus automated tests and a passing GitHub Actions CI run.
- A positive endpoint-level claim requires a paired improvement interval
  strictly above zero and an average gain of at least 5% in MAE (regression)
  or 0.02 absolute AP (classification). Apply Bonferroni correction across
  the three endpoint claims (family alpha 0.05). Otherwise say evidence is
  insufficient. A general claim of help requires all three endpoints to pass.

## Exclusions
No ligand/thesis or homelab data; no deployment; no hyperparameter search,
pretraining, graph model, prospective validation, clinical use, or SOTA claim.

## P24 continuation (2026-10-24)

The original one-split evidence remains archived in `results/`. The follow-up
in `results/robustness/` supersedes the exclusions on repeated splits and a
small hyperparameter search. Its executable plan is written before fitting.

- Five largest-first scaffold splits: 20261024–20261028; five paired model
  seeds per split: 11, 22, 33, 44, 55. Verify five distinct assignments and
  5 × 5 × 2 × 3 = 150 finite test metrics with saved predictions.
- Each approach searches the same three complete task bundles on every split:
  (learning rate, weight decay) = (0.001, 0.0001), (0.0003, 0.0001),
  (0.001, 0.001). Use search seed 101 only, same architecture, maximum 60
  epochs and patience 10. Pick one bundle per approach using the mean of
  three normalized validation losses, with first-candidate tie breaking.
  Each ST bundle fits three networks; each MT bundle fits one network.
  Equal budget means candidate/task exposure, not equal FLOPs or runtime.
- All choices are fixed for both approaches before test scoring within each
  split. Test targets are replaced by missing values for every fit and search
  score. Never select a candidate or split using test performance. Molecules
  may change roles across separate splits, as in repeated holdout.
- Report endpoint paired gains, pointwise 95% hierarchical percentile
  bootstrap intervals (20,000 draws, seed 7301), resampling split clusters
  then paired model seeds within sampled splits. Retain Bonferroni endpoint
  claim control using 98.3333% intervals and the original practical thresholds.
- Audit each test molecule's nearest training molecule in the global union,
  radius-2 1024-bit Morgan Tanimoto; persist neighbors, distributions, task
  subsets and counts at 0.8, 0.9 and 1.0. This is a descriptive audit, not a
  test-driven exclusion rule. Also report canonical-tautomer cross-partition
  identities and any truncated enumerations; do not silently alter splits.

Five overlapping repeated holdouts are not independent datasets. Largest-first
assignment still anchors large scaffold groups in training. Bootstrap intervals
are exploratory uncertainty under this split algorithm and search procedure;
external population uncertainty, broad search, graph baselines, similarity
exclusion experiments and tautomer-grouped retraining remain outside scope.

## Outcome of the continuation (executed 2026-10-24)

Executed as written: 5 splits × 5 seeds = 150/150 metrics, 30/30 search
bundles, 160 fits. Multitask was not supported on any endpoint; solubility was
worse (simultaneous interval [-0.0701, -0.0255] MAE). Still not done: broad
search, per-task independent search, graph baselines, similarity exclusion,
tautomer-grouped retraining, external validation.
