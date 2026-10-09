"""Deterministic secondary analysis; no simulation, fitting, or network calls.

Run: python application/analyze_application.py
Inputs: archived population truth and Monte Carlo summaries in correlation/results.
Outputs: application/results. Ratios are descriptive, not new utility measures.
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
from scipy.special import betaincc

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(parents=True, exist_ok=True)
TOL = 1e-12


def beta_nb(mean, var, t):
    k = mean * (1 - mean) / var - 1
    a, b = mean * k, (1 - mean) * k
    return (mean * betaincc(a + 1, b, t) - t * betaincc(a, b, t)) / (1 - t)


def population_analysis():
    pop = pd.read_csv(ROOT / 'correlation/results/population_truth.csv')
    assert np.allclose(pop.mean_a, pop.mean_b, rtol=0, atol=TOL)
    # Calibration makes the common score mean the event prevalence.
    pop['prevalence'] = pop.mean_a
    pop['nb_all'] = (pop.prevalence - pop.threshold) / (1 - pop.threshold)
    pop['best_default'] = np.maximum(0, pop.nb_all)
    pop['best_true_nb'] = pop[['true_nb_a', 'true_nb_b']].max(axis=1)
    pop['available_gain'] = pop.best_true_nb - pop.best_default
    assert (pop.available_gain >= -TOL).all()
    pop['gap_fraction_of_gain'] = np.where(
        pop.available_gain > TOL, abs(pop.true_delta) / pop.available_gain, np.nan)
    for model in ['a', 'b']:
        pop['beta_nb_' + model] = beta_nb(pop['mean_' + model], pop['var_' + model], pop.threshold)
        # Positive values are the largest common, per-evaluated-person burden
        # at which that model improves on the best all/none default.
        pop['true_gain_' + model] = pop['true_nb_' + model] - pop.best_default
        pop['beta_gain_' + model] = pop['beta_nb_' + model] - pop.best_default
    beta_error = np.max(abs(pop.beta_nb_a - pop.beta_nb_b - pop.beta_delta))
    assert beta_error < TOL
    pop['true_adoption_ceiling'] = pop.available_gain
    pop['beta_adoption_ceiling'] = pop[['beta_gain_a', 'beta_gain_b']].max(axis=1)
    pop.to_csv(OUT / 'all_population_cells.csv', index=False)
    pop[pop.primary].to_csv(OUT / 'primary_population.csv', index=False)

    summary = pd.read_csv(ROOT / 'correlation/results/summary.csv')
    # Match using a rounded numerical key; never drop a retained threshold.
    pop['threshold_key'] = pop.threshold.round(12)
    summary['threshold_key'] = summary.threshold.round(12)
    merged = summary.merge(pop[['case', 'threshold_key', 'best_default', 'available_gain']],
                           on=['case', 'threshold_key'], how='left', validate='many_to_one')
    assert len(merged) == len(summary) and merged.available_gain.notna().all()
    for col in ['mean_loss', 'loss_low', 'loss_high', 'loss_mcse']:
        merged[col + '_fraction_of_gain'] = np.where(
            merged.available_gain > TOL, merged[col] / merged.available_gain, np.nan)
    merged.drop(columns='threshold_key').to_csv(OUT / 'all_loss_fractions.csv', index=False)
    selected = merged[(merged.n == 10000) & merged.primary &
                      (merged.method == 'beta_estimated_mean') &
                      np.isclose(merged.rho0, merged.rho1) & merged.rho0.isin([0, .9])]
    selected.to_csv(OUT / 'primary_loss_fractions.csv', index=False)

    # Exact breakpoints suffice for a common burden: no outcome simulation.
    # At ties choose default, then A, then B. Ignore boundary ties in intervals.
    intervals = []
    for row in pop.itertuples():
        ceilings = [row.true_gain_a, row.true_gain_b, row.beta_gain_a, row.beta_gain_b]
        cuts = sorted(set([0.] + [float(x) for x in ceilings if x > TOL]))
        cuts.append(cuts[-1] + max(.01, cuts[-1] * .1))
        for lower, upper in zip(cuts[:-1], cuts[1:]):
            if upper - lower < TOL:
                continue
            h = (lower + upper) / 2
            truth = np.array([row.best_default, row.true_nb_a - h, row.true_nb_b - h])
            approx = np.array([row.best_default, row.beta_nb_a - h, row.beta_nb_b - h])
            true_pick, beta_pick = int(np.argmax(truth)), int(np.argmax(approx))
            intervals.append(dict(case=row.case, threshold=row.threshold, primary=row.primary,
                burden_lower=lower, burden_upper=upper, true_choice=['default','A','B'][true_pick],
                beta_choice=['default','A','B'][beta_pick],
                beta_choice_below_default=bool(truth[beta_pick] < row.best_default - TOL),
                loss_at_interval_midpoint=float(truth[true_pick] - truth[beta_pick])))
    pd.DataFrame(intervals).to_csv(OUT / 'burden_intervals.csv', index=False)
    return len(pop), len(merged), float(beta_error)


def summary_reconstruction():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a = np.array([.05, .15, .4, .8, .2, .45, .65, .95])
    aligned = np.array([.1, .25, .35, .7, .3, .4, .6, .9])
    permuted = np.r_[aligned[:4][::-1], aligned[4:][::-1]]
    fixtures = [
        ('aligned', y, a, aligned),
        ('permuted_within_outcome', y, a, permuted),
        ('ties_and_endpoints', y, np.array([0, .2, .5, 1, .2, .5, 1, 0]),
         np.array([1, .5, .2, 0, .5, .2, 0, 1])),
        ('constant_scores', y, np.full(8, .2), np.full(8, .2))]
    rows, cells = [], []
    for name, yy, aa, bb in fixtures:
        n = len(yy)
        for t in [.05, 1/9, .2, .5, .7, .95, .99]:
            da, db = aa >= t, bb >= t
            ha, hb = np.maximum(aa-t, 0)/(1-t), np.maximum(bb-t, 0)/(1-t)
            tail_a = (aa[da].sum() - t*da.sum()) / (n*(1-t))
            tail_b = (bb[db].sum() - t*db.sum()) / (n*(1-t))
            risk_d = ha - hb
            risk_mean = tail_a-tail_b
            risk_se = np.sqrt(max(0, (risk_d@risk_d - n*risk_mean**2)/(n*(n-1))))
            outcome_d = (yy-t)/(1-t)*(da.astype(int)-db.astype(int))
            s1 = s2 = 0.
            for event in [0, 1]:
                for decision_a in [0, 1]:
                    for decision_b in [0, 1]:
                        count = int(np.sum((yy == event) & (da == decision_a) & (db == decision_b)))
                        z = (event-t)/(1-t)*(decision_a-decision_b)
                        s1 += count*z
                        s2 += count*z*z
                        cells.append(dict(fixture=name, threshold=t, outcome=event,
                                          decision_a=decision_a, decision_b=decision_b, count=count))
            outcome_mean = s1/n
            outcome_se = np.sqrt(max(0, (s2-n*outcome_mean**2)/(n*(n-1))))
            errors = [abs(tail_a-ha.mean()), abs(tail_b-hb.mean()),
                      abs(risk_mean-risk_d.mean()), abs(risk_se-risk_d.std(ddof=1)/np.sqrt(n)),
                      abs(outcome_mean-outcome_d.mean()), abs(outcome_se-outcome_d.std(ddof=1)/np.sqrt(n))]
            assert max(errors) < 1e-10
            rows.append(dict(fixture=name, threshold=t, n=n, risk_contrast=risk_mean,
                             risk_se=risk_se, outcome_contrast=outcome_mean, outcome_se=outcome_se,
                             maximum_absolute_error=max(errors)))
    frame = pd.DataFrame(rows)
    demo = frame[(frame.threshold == .5) & frame.fixture.isin(['aligned','permuted_within_outcome'])]
    assert np.allclose(demo.outcome_contrast, 0)
    assert demo.iloc[0].outcome_se == 0 and demo.iloc[1].outcome_se > 0
    frame.to_csv(OUT / 'summary_reconstruction.csv', index=False)
    pd.DataFrame(cells).to_csv(OUT / 'joint_eight_cell_counts.csv', index=False)
    demo.to_csv(OUT / 'same_marginals_different_pairing.csv', index=False)
    return len(frame), float(frame.maximum_absolute_error.max())


def main():
    population_cells, summary_rows, beta_error = population_analysis()
    fixture_cells, reconstruction_error = summary_reconstruction()
    report = dict(status='PASS', population_cells=population_cells, summary_rows=summary_rows,
                  fixture_threshold_cells=fixture_cells, max_beta_reconstruction_error=beta_error,
                  max_summary_reconstruction_error=reconstruction_error,
                  scope='Secondary deterministic analysis and algebra checks; no new random samples, model training, coverage evaluation, or clinical validation.',
                  python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                  inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [ROOT/'correlation/results/population_truth.csv',
                                    ROOT/'correlation/results/summary.csv']})
    (OUT/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    print(pd.read_csv(OUT/'primary_population.csv')[['case','available_gain','true_gain_a','true_gain_b','beta_gain_a','beta_gain_b']].to_string(index=False))
    print(pd.read_csv(OUT/'primary_loss_fractions.csv')[['case','rho0','mean_loss_fraction_of_gain']].to_string(index=False))
    print(pd.read_csv(OUT/'same_marginals_different_pairing.csv').to_string(index=False))


if __name__ == '__main__':
    main()
