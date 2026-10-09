"""Optional full historical design. Not run by the default verifier."""
# Kaggle: CPU only; no Internet or uploaded files required.
# Synthetic finite-sample audit, NOT external clinical validation.
from pathlib import Path
from datetime import datetime, timezone
import json, time, platform, zipfile
import numpy as np
import pandas as pd
import scipy
from scipy.stats import beta
from scipy.integrate import quad

REPS = 200
SIZES = [500, 3000, 10000]
TS = np.array([.10, .15, .20, .25, .30, .40])
SEED = 2026100844
MU = .2
TOL = 1e-10
CASES = ['beta_control', 'mixture_025', 'mixture_050']
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'verified_outputs/finite_full'
P=json.loads((BASE/'protocols/finite.json').read_text())
assert P['seed']==SEED and P['repetitions']==REPS
assert P['sample_sizes']==SIZES and P['cases']==CASES
assert np.array_equal(P['thresholds'],TS)

def matched(mu, var):
    if not 0 < var < mu * (1-mu):
        raise ValueError('Invalid Beta moments; do not silently clip.')
    k = mu * (1-mu) / var - 1
    return np.array([mu*k]), np.array([(1-mu)*k]), np.array([1.])

def moments(par):
    a, b, w = par
    m = float(w @ (a/(a+b)))
    v = float(w @ (a*(a+1)/((a+b)*(a+b+1))) - m*m)
    return m, v

def population_nb(par):
    a, b, w = par
    t = TS[:, None]
    terms = a/(a+b)*beta.sf(t, a+1, b) - t*beta.sf(t, a, b)
    return (terms @ w)/(1-TS)

def make_mixture(kind, lam):
    k = 1000.
    v = .01 if kind == 'A' else .015
    vc = ((k+1)*v - MU*(1-MU))/k
    if kind == 'A':
        c = np.array([MU-np.sqrt(vc), MU+np.sqrt(vc)])
        w = np.array([.5, .5])
    else:
        lo, hi = .02, .8
        c = np.array([lo, MU, hi])
        wl = vc/((MU-lo)*(hi-lo))
        wh = vc/((hi-MU)*(hi-lo))
        w = np.array([wl, 1-wl-wh, wh])
    endpoints = np.array([.001, .999])
    high_weight = (MU-.001)/(.999-.001)
    c = np.r_[c, endpoints]
    w = np.r_[(1-lam)*w, lam*np.array([1-high_weight, high_weight])]
    par = c*k, (1-c)*k, w
    assert np.all(w >= 0) and abs(w.sum()-1) < 1e-12
    assert abs(moments(par)[0]-MU) < 1e-12
    return par

def calibrated_auc(par):
    a, b, w = par
    m, _ = moments(par)
    def integrand(s):
        f = float(w @ beta.cdf(s, a, b))
        return f*(1-f)
    val, err = quad(integrand, 0, 1, points=sorted(set(a/(a+b))),
                    epsabs=1e-10, limit=500)
    assert err < 1e-7
    return .5 + val/(2*m*(1-m))

def sample_given_y(par, y, rng):
    # Bayes construction: both models are calibrated on the SAME population.
    # f(s|Y=1)=s*f(s)/mu; f(s|Y=0)=(1-s)*f(s)/(1-mu).
    a, b, w = par
    m, _ = moments(par)
    s = np.empty(len(y))
    for label in [0, 1]:
        ids = np.flatnonzero(y == label)
        weights = w*(a/(a+b)/m if label else b/(a+b)/(1-m))
        assert abs(weights.sum()-1) < 1e-10
        component = rng.choice(len(w), size=len(ids), p=weights/weights.sum())
        s[ids] = rng.beta(a[component]+label, b[component]+1-label)
    return s

def beta_estimate(s, mu=None):
    return population_nb(matched(float(s.mean()) if mu is None else mu,
                                 float(s.var(ddof=0))))

def wrong(delta, truth):
    return ((delta > TOL) & (truth < -TOL)) | ((delta < -TOL) & (truth > TOL))

def wilson(x):
    n = len(x); p = float(np.mean(x)); z = 1.959963984540054
    den = 1+z*z/n
    mid = (p+z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return p, max(0., mid-half), min(1., mid+half)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    protocol = dict(reps=REPS, sample_sizes=SIZES, thresholds=TS.tolist(),
        seed=SEED, cases=CASES, prevalence=MU, primary_threshold=.2,
        status='Exploratory finite-sample supplement to an existing counterexample',
        calibration='Exact population calibration by shared-outcome Bayes construction',
        dependence='Scores conditionally independent given Y; one chosen joint law',
        variance_ddof=0, inference='Pointwise paired Wald interval for empirical NB difference',
        limits=['Artificial populations; not clinical failure frequencies',
                'No Stata execution; analytic expectation of the standard Beta procedure',
                'No novel estimator or novelty guarantee',
                'Monte Carlo rate intervals describe only this fixed simulation'])
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
    rows, population, quality = [], [], []
    for ci, case in enumerate(CASES):
        if case == 'beta_control':
            pa, pb = matched(MU, .01), matched(MU, .015)
        else:
            lam = .25 if case == 'mixture_025' else .50
            pa, pb = make_mixture('A', lam), make_mixture('B', lam)
        ta, tb = population_nb(pa), population_nb(pb)
        ma, va = moments(pa); mb, vb = moments(pb)
        ba = population_nb(matched(ma, va))
        bb = population_nb(matched(mb, vb))
        truth = ta-tb
        assert np.all(abs(truth) > TOL), 'Population tie: revise tie reporting before running.'
        if case == 'beta_control':
            assert np.max(abs(truth-(ba-bb))) < 1e-12
        for model, par in [('A', pa), ('B', pb)]:
            m, v = moments(par)
            quality.append(dict(case=case, model=model, mean=m, variance=v,
                                auc=calibrated_auc(par), brier=m*(1-m)-v))
        for j, t in enumerate(TS):
            population.append(dict(case=case, threshold=t, true_A=ta[j], true_B=tb[j],
                true_delta=truth[j], beta_A=ba[j], beta_B=bb[j], beta_delta=ba[j]-bb[j],
                population_beta_wrong=bool(wrong(ba-bb, truth)[j])))
        for ni, n in enumerate(SIZES):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, ci, ni]))
            for rep in range(REPS):
                y = rng.binomial(1, MU, n)
                a = sample_given_y(pa, y, rng)
                b = sample_given_y(pb, y, rng)
                ia = a[:, None] >= TS
                ib = b[:, None] >= TS
                contributions = (ia.astype(int)-ib.astype(int))*(y[:, None]-TS)/(1-TS)
                observed = contributions.mean(axis=0)
                se = contributions.std(axis=0, ddof=1)/np.sqrt(n)
                low, high = observed-1.959963984540054*se, observed+1.959963984540054*se
                proxy = (np.maximum(a[:, None]-TS, 0)-np.maximum(b[:, None]-TS, 0)).mean(axis=0)/(1-TS)
                estimated = beta_estimate(a)-beta_estimate(b)
                # Diagnostic only: use known common population mean to isolate mean noise.
                known_mu = beta_estimate(a, MU)-beta_estimate(b, MU)
                bw, ow = wrong(estimated, truth), wrong(observed, truth)
                ci_correct = ((low > 0) & (truth > 0)) | ((high < 0) & (truth < 0))
                for j, t in enumerate(TS):
                    rows.append(dict(case=case, n=n, rep=rep, threshold=t,
                        true_delta=truth[j], estimated_beta_delta=estimated[j],
                        observed_delta=observed[j], proxy_delta=proxy[j],
                        beta_wrong=bool(bw[j]), observed_wrong=bool(ow[j]),
                        known_mu_beta_wrong=bool(wrong(known_mu, truth)[j]),
                        beta_tie=bool(abs(estimated[j]) <= TOL),
                        observed_tie=bool(abs(observed[j]) <= TOL),
                        beta_population_regret=abs(truth[j])*bw[j],
                        observed_population_regret=abs(truth[j])*ow[j],
                        observed_ci_low=low[j], observed_ci_high=high[j],
                        observed_ci_covers=bool(low[j] <= truth[j] <= high[j]),
                        beta_wrong_ipd_ci_correct=bool(bw[j] and ci_correct[j])))
            print(f'{case}: n={n}, {REPS}/{REPS} repetitions finished', flush=True)
    d = pd.DataFrame(rows); summaries = []
    for keys, g in d.groupby(['case', 'n', 'threshold'], sort=False):
        r = dict(zip(['case', 'n', 'threshold'], keys)); r['repetitions'] = len(g)
        for col in ['beta_wrong', 'observed_wrong', 'known_mu_beta_wrong',
                    'beta_wrong_ipd_ci_correct']:
            rate, lo, hi = wilson(g[col].to_numpy())
            r[col+'_rate'], r[col+'_mc_lo'], r[col+'_mc_hi'] = rate, lo, hi
        for col in ['estimated_beta_delta', 'observed_delta', 'proxy_delta',
                    'beta_population_regret', 'observed_population_regret',
                    'observed_ci_covers', 'beta_tie', 'observed_tie']:
            r[col+'_mean'] = float(g[col].mean())
        summaries.append(r)
    s = pd.DataFrame(summaries); pop = pd.DataFrame(population); q = pd.DataFrame(quality)
    assert len(d) == len(CASES)*len(SIZES)*REPS*len(TS)
    assert not d.duplicated(['case', 'n', 'rep', 'threshold']).any()
    for name, data in [('replicates', d), ('summary', s), ('population', pop), ('population_quality', q)]:
        data.to_csv(OUT/(name+'.csv'), index=False)
    (OUT/'metadata.json').write_text(json.dumps(dict(python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
        elapsed_seconds=time.perf_counter()-start), indent=2), encoding='utf8')
    archive = OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in OUT.iterdir():
            if f.is_file(): z.write(f, f.name)
    print('\nPopulation model quality:\n', q.to_string(index=False))
    print('\nPopulation results at t=0.20:\n', pop[np.isclose(pop.threshold,.2)].to_string(index=False))
    cols = ['case','n','beta_wrong_rate','beta_wrong_mc_lo','beta_wrong_mc_hi',
            'observed_wrong_rate','known_mu_beta_wrong_rate','beta_population_regret_mean']
    print('\nFinite-sample primary results:\n', s.loc[np.isclose(s.threshold,.2),cols].to_string(index=False))
    print('\nResults ZIP:', archive)
    try:
        from IPython.display import display, FileLink
        display(FileLink(str(archive)))
    except ImportError:
        pass
    return s

if __name__ == '__main__':
    result = main()
