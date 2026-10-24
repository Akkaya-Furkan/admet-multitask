# Experimental protocol

The frozen executable settings are `results/config.json`. Molecules are united
across the tasks before a single 70/15/15 target split. Whole Murcko scaffold
groups are assigned largest first; group-size ties are shuffled with the fixed
split seed. Empty scaffolds stay together. This design prevents a held-out
molecule or scaffold from appearing in any task's training labels. Parent
identity and scaffold disjointness are asserted, not assumed.

Inputs are binary radius-2 Morgan fingerprints. Each network has a
1024 → 128 → 64 ReLU trunk and scalar linear heads. Single-task training
activates one head and the shared-trunk model activates all three heads.
For a given seed, initial trunks and corresponding heads are identical.
The independent approach has three separate trunks, so total parameter
budgets are different; the per-task architecture is matched.

Regression loss is MSE on targets centered and scaled using training labels
only. Classification uses binary cross entropy with logits; sigmoid converts
test logits to probabilities. There is no imputation of missing targets.
Every epoch partitions each task's observed training labels into the same
number of batches (ceil of the largest task's training count / 256). Every
observed label is visited once per epoch. Per-task permutations are identical
between approaches for a seed/epoch/task. Multitask gradients use the mean of
three task losses; independent models use their individual task loss.

Adam uses learning rate 0.001 and weight decay 0.0001. Training runs for at
most 60 epochs with patience 10, restoring the lowest validation-loss state.
Single-task stopping minimizes its task's validation loss; multitask stopping
minimizes the equally weighted mean of three validation losses. No test values
are passed to the stopping criterion. Tests verify that changing test labels
cannot change trained parameters. Checkpoints are not retained; configurations,
validation histories, identities and test predictions are retained.

Each endpoint reports five paired initialization seeds on the same split:
11, 22, 33, 44, 55. Error bars are sample standard deviations (ddof=1), not
confidence intervals. A paired gain is ST MAE − MT MAE for regressions and
MT AP − ST AP for classification. Two-sided t intervals use df=4 and
Bonferroni family alpha=0.05 across the three endpoints (confidence
1 − 0.05/3 each). Positive evidence requires the whole interval above zero
and mean relative MAE reduction ≥0.05, or absolute AP increase ≥0.02.
No common scalar metric is averaged across incompatible endpoints.

These intervals only quantify training-seed variability conditional on one
split and one dataset. Normality of five differences cannot be established;
these are exploratory diagnostics, not independent experimental replication.
