# Decision context and reporting summaries

Run from the repository root: `python run.py verify-application`.
Alternatively: `python application/analyze_application.py`.
Requires NumPy, pandas and SciPy, already included in requirements-audit.txt.
No internet, GPU, patient data, random simulation, or fitted clinical models are used.

Inputs are the archived correlation/results/population_truth.csv and summary.csv.
Results retain all 16 population cells and all 1,344 method/threshold/size/dependence summaries.
primary_loss_fractions.csv displays the fixed primary thresholds at n=10,000 and rho=0 or 0.9;
other settings remain in all_loss_fractions.csv. Ratios use known population denominators;
their scaled Monte Carlo intervals are not clinical confidence intervals.

burden_intervals.csv uses exact positive breakpoints of true and Beta gains above the best
all/none default. Rows describe open intervals; at ties the declared rule favors default,
then A, then B. The final interval extends beyond all adoption ceilings; choices remain
default for every larger burden. Midpoint losses are illustrative, not averages over a
burden distribution. A common per-person burden does not alter A/B ordering.

summary_reconstruction.csv checks direct calculations against tail summaries and eight-cell
outcome/decision tables on 28 deterministic fixture-threshold cells. The two paired examples
have identical model-specific outcome/score margins but different paired standard errors.
Fixtures are not calibrated populations, clinical observations, or coverage experiments.

These are applications of established utility and inference principles, not new methods.
Third-party articles and separate author supplement packages are not redistributed.


## Binned information: completed deterministic extension
Run `python run.py verify-binning` or `python application/verify_binning.py`.
NumPy and pandas are sufficient. The script uses four fixed eight-score arrays, ten
thresholds, and four nested partitions, plus two pairs of small counterexamples.
Outputs in binning_results preserve all 160 cells, 80 aligned cells and 120 nested
refinement checks. They are deterministic checks, not Monte Carlo or coverage results.
The within-bin square sums in example_bin_inputs.csv deliberately differ in the
variance example; only bin counts and first risk sums are equal.
The checks apply classical convex bounds and sample-variance identities. They do
not constitute new inference methods, empirical clinical evidence or a guaranteed
reporting standard. Total n is assumed known for all reconstruction formulas.
