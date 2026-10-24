# Data provenance and interpretation

The downloader uses TDC's Harvard Dataverse file IDs, taken from the
[TDC metadata registry](https://github.com/mims-harvard/TDC/blob/main/tdc/metadata.py).
The exact URLs, sizes, SHA-256 hashes and parsed row counts are recorded in
`results/data_manifest.json`. The downloader verifies those frozen hashes on
both fresh downloads and cached files; a changed source fails closed.
Raw downloads are excluded from Git; standardized identities, labels, split
membership and predictions are included for auditability.

Endpoints are aqueous solubility (AqSolDB; log molar concentration), Caco-2
permeability (Wang; source log Papp scale) and CYP2D6 inhibition (Veith; binary).
See the [TDC endpoint descriptions and original-paper references](https://tdcommons.ai/single_pred_tasks/adme/).
These are heterogeneous assays, not a matched experimental panel. Unknown
labels stay missing and are never treated as negative labels.

[TDC's Dataverse record](https://doi.org/10.7910/DVN/21LKWG) advertises CC0;
the endpoint page identifies AqSolDB as CC BY 4.0 and uses ambiguous
“Not Specified / CC BY 4.0” wording for Wang and Veith. Upstream attribution
and terms should be checked before reuse; this repository does not relicense
the underlying measurements.

Standardization uses RDKit Cleanup, FragmentParent and neutralization, followed
by non-isomeric canonical SMILES. This merges salts and stereoisomers but does
not enumerate tautomers. Regression duplicates are averaged; binary-label
conflicts are removed for that task only. No label values inform split
assignment. All acyclic molecules have the same empty Murcko scaffold and
remain together. The split is custom and is not a TDC leaderboard split.
Fingerprint similarity between different scaffolds is not prohibited.
