# Beta approximation and model ranking: reproducibility archive

Companion to *Model ranking after Beta approximation in summary-data decision curve analysis*.
This archive provides the analysis code, fixed protocols, synthetic results and aggregate public-cohort results supporting the manuscript. It does not contain patient-level source data, individual predictions, the manuscript or third-party papers.

The 2026-10-09 packaging revision leaves every analysis script and archived result unchanged. See RELEASE_NOTES.md for the exact scope. The package was prepared locally for repository upload; no public repository URL or DOI is asserted by this archive.

## Check the download without running experiments
From this directory, run `python verify_manifest.py`. This standard-library-only command checks SHA-256 hashes and writes no files. It does not simulate patients, replay repetitions, fit models or download data. The archived `verification.json` is a historical analysis-verification record, not the result of this packaging-only check.

For upload instructions in Chinese, see UPLOAD_GITHUB_CN.txt. Upload the contents of this directory as the repository root, not the outer ZIP as the only repository file. Include the supplied `.gitignore` and `.gitattributes` files.

## Install
Use Python 3.12. The pinned environment records the successful current reproduction, not necessarily the historical Kaggle environment.

```bash
python -m pip install -r requirements.txt
python run.py verify
python run.py figures
```

For the offline audit and figures only, requirements-audit.txt is sufficient.
Run commands from the extracted DCA_GitHub directory. The verify command checks input hashes, recomputes archived synthetic summaries, checks formula identities, and replays only the first repetition in each of 9 original and 114 shape configurations. It does not rerun all simulations or fit clinical models. Its report is verification.json.
The shipped results represent 1,800 original and 57,000 shape repetitions, plus 31,500 paired dependence-sensitivity datasets sharing 4,500 base random streams. Replaying selected repetitions does not independently certify every historical random draw.

## Completed dependence extension
See correlation/README.md for the fixed design, execution environment and limitations. Run `python run.py verify-correlation` for saved-array verification alone, or `python run.py figures-correlation` to regenerate Figure S4. The default `verify` also checks these summaries. Full simulation is optional via `python run.py simulate-correlation`.

## Deterministic decision-context extension
Run `python run.py verify-application` for Supplement S12 and Table S6.
This secondary analysis adds default-policy gains, hypothetical common implementation burdens,
and exact recovery of fixed-threshold estimates and paired sample standard errors from summaries.
It uses archived results plus 28 deterministic checks; no new patients are simulated.
See application/README.md for the inputs, outputs, assumptions, and limitations.
Run `python run.py verify-binning` for the additional S12 bin-summary checks and Table S7.
This checks exact point recovery versus missing variance or covariance without new random draws.

## Optional full reproduction
These commands require additional time. They are not part of the default audit.
```bash
python run.py simulate-original
python run.py simulate-shape
```
Original outputs go to core/verified_outputs/finite_full; shape outputs go to reruns/shape. Preserved reference results are not overwritten.

## Optional public-cohort refitting
Read DATA_SOURCES.txt and the original source conditions first. Internet is required only for the explicit download step.
```bash
python run.py fetch-data
python run.py clinical
```
Downloaded bytes must match protocol hashes; the utility stops if upstream content changes. A download failure does not invalidate the offline simulation audit.
Clinical results are written to core/verified_outputs, including patient-level predictions; do not upload this ignored directory. Compare aggregate estimates with reference_clinical. Re-fitting can differ with library/platform changes. Full clinical refits were not rerun merely to assemble this release.
The clinical command also recomputes the GUSTO error decomposition and binned bounds. Original protocol files specify splits, seeds, features, thresholds, calibration and bootstrap settings. These are exploratory methodological illustrations, not confirmatory external clinical validation.

## Files
- core/analysis: original simulation generator, fixed clinical refits and postprocessing.
- core/protocols: fixed analysis specifications.
- core/inputs/finite: synthetic repeated estimates and reference summaries.
- dca_shape_audit: fixed-moment shape construction and simulation generator.
- dca_round_final/results: archived population grid and 114 synthetic result arrays.
- dca_round_final/analyze.py: independent summary recomputation and first-repetition replay.
- reference_clinical: aggregate public-cohort results only.
- scripts: the two main manuscript figure generators.
- SHA256SUMS.json: input/code integrity manifest.
- PROVENANCE.json: original archive hash and packaging changes.

The original clinical source datasets, individual predictions, split membership, third-party articles and author software supplement archives are not distributed. No patient-level files should be added to GitHub.

## Optional deterministic theoretical checks
The theory directory retains the implementations and saved audit results used for the source-formula, weighted-Brier perturbation, sharp-bound and ranking-boundary checks. These are not claimed as new classical bounds. Source hashes identify external materials; the external papers and software are not redistributed.
```bash
python theory/check_source_formulas.py
python theory/check_prior_example_perturbation.py
python theory/check_sharp_bounds.py
python theory/check_ranking_boundary.py
python dca_shape_audit/verify_population.py
```
These commands are separate from the bounded default audit. Some print JSON to standard output. The population check uses independent integration; the boundary check also writes a figure.

## Review and public release
For double-anonymous review, upload this ZIP through the journal supplement/reviewer-file mechanism permitted by the editor. Do not send the internal manuscript-building ZIP as the anonymous reproducibility supplement. A personal GitHub account can identify the authors; use a journal attachment for blinded review.
To prepare GitHub, extract this folder and upload its contents. Do not invent a URL in the paper before the actual repository exists. After publication, cite the actual URL and record a commit or release for the manuscript. A private repository is not accessible to readers through an ordinary link. This archive can also be supplied as a journal attachment for review. No DOI has been assigned within this package.

The ignore rules protect normal Git-based workflows, not manual browser uploads. Do not upload downloaded or regenerated patient-level files even if using the website. `.gitattributes` disables automatic line-ending conversion to preserve checksummed bytes. No blanket software or data license is granted here; the rights holders should select any public reuse license separately, without relicensing third-party datasets.

## Scope and limitations
The contribution concerns model-ranking sensitivity of a two-moment Beta approximation. Classical moment bounds and calibrated simulation identities are not claimed as new. High-discrimination and low-endpoint-mass examples are distinct. Neither public cohort established opposite ordering under its uncertainty diagnostic.
The conditional-copula extension has now been run and is archived under correlation/. Its audit recalculates saved estimates without rerunning simulations. It covers one copula family and three selected pairs, not all dependence structures.

## AI assistance
ChatGPT (OpenAI) assisted with design, mathematical work, coding, checks, interpretation and writing. The study experiments were executed in Python on Kaggle. Kaggle is a computing platform, not a generative AI tool. Human authors remain responsible for the submitted work. Exact model versions were not consistently retained.
