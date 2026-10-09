"""Conditional-copula sensitivity audit; CPU only, no external data.
Adapted simulation principle: Pfeiffer & Gail (2020), Section 4.3.
The full experiment has NOT been run by the preparer.
"""
from pathlib import Path
import json, time, sys, hashlib, shutil
import numpy as np
import pandas as pd
import scipy
from scipy.special import betainc, betaincc, ndtr, expit
import matplotlib.pyplot as plt

VERSION = "dca-copula-1.0"
MU, TOL, SEED = .2, 1e-10, 2026100829
METHODS = ["beta_estimated_mean", "risk_only", "outcome_based", "beta_known_mean"]
RHOS = [(-.5, -.5), (0., 0.), (.3, .3), (.6, .6), (.9, .9), (0., .9), (.9, 0.)]
BASELINE = RHOS.index((0., 0.))

def mix(w, a, b):
    w, a, b = map(lambda x: np.asarray(x, float), (w, a, b))
    keep = w > 0
    assert np.all(w >= 0) and abs(w.sum()-1) < 1e-12
    assert np.all(a > 0) and np.all(b > 0)
    return w[keep]/w[keep].sum(), a[keep], b[keep]

def moments(c):
    w, a, b = c
    m = np.sum(w*a/(a+b))
    return m, np.sum(w*a*(a+1)/((a+b)*(a+b+1)))-m*m

def shape(v, s, eps):
    k, vmax = 50., MU*(1-MU)
    vc = ((k+1)*v-vmax)/k
    lo, hi = vc/(vc+MU**2), (1-MU)**2/((1-MU)**2+vc)
    w = lo+s*(hi-lo)
    centers = np.array([MU-np.sqrt(vc*(1-w)/w), MU+np.sqrt(vc*w/(1-w))])
    q = vmax/v-1
    return mix([1-eps, eps*w, eps*(1-w)],
               np.r_[MU*q, k*centers], np.r_[(1-MU)*q, k*(1-centers)])

def high_auc(is_a):
    k, v = 1000., (.01 if is_a else .015)
    vc = ((k+1)*v-MU*(1-MU))/k
    if is_a:
        c, w = [MU-np.sqrt(vc), MU+np.sqrt(vc)], [.5, .5]
    else:
        wl, wh = vc/((MU-.02)*(.8-.02)), vc/((.8-MU)*(.8-.02))
        c, w = [.02, MU, .8], [wl, 1-wl-wh, wh]
    h = (MU-.001)/(.999-.001)
    c, w = np.r_[c, .001, .999], np.r_[.5*np.array(w), .5*(1-h), .5*h]
    return mix(w, k*c, k*(1-c))

def cases():
    odds = np.array([.25, .5, 1., 2., 4.])*MU/(1-MU)
    t = odds/(1+odds)
    return [("beta_control", shape(.008, .5, 0), shape(.0088, .5, 0), t, .2),
            ("low_endpoint", shape(.008, .1, .5), shape(.0088, .5, .5), t, 1/9),
            ("high_auc", high_auc(True), high_auc(False),
             np.array([.1, .15, .2, .25, .3, .4]), .2)]

def nb(c, t):
    w, a, b = c
    x = np.asarray(t)[:, None]
    return ((w*(a/(a+b)*betaincc(a+1, b, x)-x*betaincc(a, b, x))).sum(1)/(1-t))

def beta_nb(m, v, t):
    if not 0 < v < m*(1-m):
        return np.full(len(t), np.nan)
    k = m*(1-m)/v-1
    return nb(mix([1.], [m*k], [(1-m)*k]), t)

def conditional(c, y):
    w, a, b = c
    p = w*(a if y else b)/(a+b)/(MU if y else 1-MU)
    assert abs(p.sum()-1) < 1e-12
    return mix(p, a+y, b+1-y)

def cdf(c, x):
    w, a, b = c
    return sum(ww*betainc(aa, bb, x) for ww, aa, bb in zip(w, a, b))

def quantile_table(c):
    # Numerical inverse: adaptive linear interpolation, audited in CDF units.
    # The audit is on a dense test set, not a proved global error bound.
    for size in [16385, 32769, 65537, 131073, 262145, 524289]:
        x = np.unique(np.r_[0., expit(np.linspace(-32, 32, size)),
                            np.linspace(0, 1, size), 1.])
        u = np.maximum.accumulate(np.clip(cdf(c, x), 0, 1))
        uq, idx = np.unique(u, return_index=True)
        xq = x[idx]; xq[0], xq[-1] = 0., 1.
        tests = np.unique(np.r_[uq[:-1]+.25*np.diff(uq),
                               uq[:-1]+.5*np.diff(uq),
                               uq[:-1]+.75*np.diff(uq),
                               np.linspace(0, 1, 10001),
                               np.geomspace(1e-12, .1, 2000),
                               1-np.geomspace(1e-12, .1, 2000)])
        error = float(np.max(np.abs(cdf(c, np.interp(tests, uq, xq))-tests)))
        if error <= 1e-7:
            return (uq, xq), dict(grid_size=size, knots=len(uq),
                                 tested_points=len(tests), max_cdf_error=error)
    raise RuntimeError("Quantile audit failed; do not interpret simulation outputs.")

def sample(tables, y, u):
    s = np.empty(len(y))
    for yy in (0, 1):
        ix = y == yy
        s[ix] = np.interp(u[ix], *tables[yy])
    return s

def estimates(s, y, t):
    m, v = s.mean(), s.var(ddof=0)
    return np.column_stack([beta_nb(m, v, t),
        np.maximum(s[:, None]-t, 0).mean(0)/(1-t),
        ((y[:, None]-t)*(s[:, None] >= t)).mean(0)/(1-t),
        beta_nb(MU, v, t)])

def diagnostics(a, b, y):
    def corr(mask):
        return np.corrcoef(a[mask], b[mask])[0, 1] if mask.sum() > 2 else np.nan
    return [corr(np.ones(len(y), bool)), corr(y == 0), corr(y == 1),
            a.mean(), b.mean(), a.var(), b.var(),
            (y-a).mean(), (y-b).mean(), np.mean((y-a)*a), np.mean((y-b)*b)]

DIAGS = ["corr_all", "corr_y0", "corr_y1", "mean_a", "mean_b", "var_a", "var_b",
         "cal_resid_a", "cal_resid_b", "cal_score_resid_a", "cal_score_resid_b"]

def avg_se(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return (float(x.mean()), float(x.std(ddof=1)/np.sqrt(len(x)))) if len(x)>1 else (np.nan, np.nan)

def wilson(p, n):
    z = 1.959963984540054
    d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h, c+h

def scores(e, truth):
    tie = np.abs(e) <= TOL
    wrong = ((e*truth < 0) & ~tie).astype(float)
    loss = np.where(e >= -TOL, max(0., -truth), max(0., truth))
    return wrong, tie.astype(float), loss

def run(reps=500, sizes=(500, 3000, 10000), output=None):
    assert reps >= 2
    base = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
    out = Path(output) if output else base/'DCA_correlation_full'
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes() if '__file__' in globals() and Path(__file__).is_file() else None
    cfg = dict(version=VERSION, reps=reps, sizes=list(sizes), seed=SEED,
               rho_pairs=RHOS, methods=METHODS, tolerance=TOL, tie_rule="choose A",
               numpy=np.__version__, scipy=scipy.__version__, python=sys.version,
               marginal_variance="ddof=0", quantile_test_tolerance=1e-7,
               source_sha256=hashlib.sha256(source).hexdigest() if source else None,
               cases=[dict(name=q[0], A=[x.tolist() for x in q[1]],
                           B=[x.tolist() for x in q[2]], thresholds=q[3].tolist(), primary=q[4]) for q in cases()])
    cfg = json.loads(json.dumps(cfg))
    path = out/'run_config.json'
    if path.exists():
        assert json.loads(path.read_text()) == cfg, "Different configuration: use a new output folder."
    path.write_text(json.dumps(cfg, indent=2))
    rows, changes, checks, pop, audit = [], [], [], [], []
    start = time.perf_counter()
    for sid, (name, ca, cb, ts, primary) in enumerate(cases()):
        na, nb_true = nb(ca, ts), nb(cb, ts)
        delta = na-nb_true
        ma, va = moments(ca); mb, vb = moments(cb)
        assert max(abs(ma-MU), abs(mb-MU)) < 1e-12
        bd = beta_nb(ma, va, ts)-beta_nb(mb, vb, ts)
        for j, t in enumerate(ts):
            pop.append(dict(case=name, threshold=t, primary=bool(abs(t-primary)<1e-12),
                            true_nb_a=na[j], true_nb_b=nb_true[j], true_delta=delta[j],
                            beta_delta=bd[j], mean_a=ma, mean_b=mb, var_a=va, var_b=vb,
                            baseline_corr_theory=np.sqrt(va*vb)/(MU*(1-MU))))
        tables = []
        for label, c in zip(['A', 'B'], [ca, cb]):
            ct = []
            for yy in (0, 1):
                table, report = quantile_table(conditional(c, yy))
                ct.append(table); audit.append(dict(case=name, model=label, y=yy, **report))
            tables.append(ct)
        pd.DataFrame(audit).to_csv(out/'quantile_audit.csv', index=False)
        print(f'{name}: inverse-CDF audit passed; elapsed {time.perf_counter()-start:.1f}s', flush=True)
        for n in sizes:
            checkpoint = out/f'{name}_n{n}.npz'
            if checkpoint.exists():
                with np.load(checkpoint) as z:
                    est, diag = z['estimates'], z['diagnostics']
                    assert est.shape == (reps, len(RHOS), len(ts), 4)
                    assert np.allclose(z['truth'], delta, atol=1e-14, rtol=0)
            else:
                est = np.empty((reps, len(RHOS), len(ts), 4))
                diag = np.empty((reps, len(RHOS), len(DIAGS)))
                for r in range(reps):
                    rng = np.random.default_rng(np.random.SeedSequence([SEED, sid, n, r]))
                    y = rng.binomial(1, MU, n)
                    za, z0 = rng.standard_normal((2, n))
                    a = sample(tables[0], y, ndtr(za))
                    ea = estimates(a, y, ts)
                    for k, (r0, r1) in enumerate(RHOS):
                        rho = np.where(y == 1, r1, r0)
                        b = sample(tables[1], y, ndtr(rho*za+np.sqrt(1-rho*rho)*z0))
                        est[r, k] = ea-estimates(b, y, ts)
                        diag[r, k] = diagnostics(a, b, y)
                    if (r+1) % 100 == 0:
                        print(f'  {name}, n={n}: {r+1}/{reps}', flush=True)
                temp = checkpoint.with_suffix('.tmp.npz')
                np.savez_compressed(temp, estimates=est, diagnostics=diag, truth=delta, thresholds=ts)
                temp.replace(checkpoint)
            for k, (r0, r1) in enumerate(RHOS):
                key = dict(case=name, n=n, rho0=r0, rho1=r1)
                for q, field in enumerate(DIAGS):
                    mean, se = avg_se(diag[:, k, q])
                    checks.append(dict(**key, diagnostic=field, mean=mean, mcse=se))
                for j, t in enumerate(ts):
                    for m, method in enumerate(METHODS):
                        e, ref = est[:, k, j, m], est[:, BASELINE, j, m]
                        valid = np.isfinite(e); ev = e[valid]; nv = len(ev)
                        key2 = dict(**key, threshold=t, primary=bool(abs(t-primary)<1e-12), method=method)
                        if nv < 2:
                            raise RuntimeError(f'Insufficient valid repetitions: {key2}')
                        wrong, ties, loss = scores(ev, delta[j])
                        rate, _ = avg_se(wrong); low, high = wilson(rate, nv)
                        selrate = float(np.mean(loss > 0))
                        sello, selhi = wilson(selrate, nv)
                        lossmean, lossse = avg_se(loss); bias, biasse = avg_se(ev-delta[j])
                        rows.append(dict(**key2, true_delta=delta[j], valid_reps=nv, failed_reps=reps-nv,
                            strict_wrong_rate=rate, wrong_low=low, wrong_high=high, tie_rate=ties.mean(),
                            selection_error_rate=selrate,
                            loss_low=abs(delta[j])*sello, loss_high=abs(delta[j])*selhi,
                            mean_loss=lossmean, loss_mcse=lossse, bias=bias, bias_mcse=biasse,
                            rmse=np.sqrt(np.mean((ev-delta[j])**2)), estimate_sd=ev.std(ddof=1)))
                        paired = valid & np.isfinite(ref)
                        w1, _, l1 = scores(e[paired], delta[j]); w0, _, l0 = scores(ref[paired], delta[j])
                        dw, sew = avg_se(w1-w0); dl, sel = avg_se(l1-l0)
                        changes.append(dict(**key2, paired_reps=int(paired.sum()),
                            wrong_change=dw, wrong_change_mcse=sew, loss_change=dl, loss_change_mcse=sel))
            pd.DataFrame(rows).to_csv(out/'summary.csv', index=False)
            pd.DataFrame(changes).to_csv(out/'paired_changes.csv', index=False)
            pd.DataFrame(checks).to_csv(out/'diagnostics.csv', index=False)
            print(f'Saved {name}, n={n}; elapsed {time.perf_counter()-start:.1f}s', flush=True)
    pd.DataFrame(pop).to_csv(out/'population_truth.csv', index=False)
    result = pd.DataFrame(rows); primary_rows = result[result.primary]
    primary_rows.to_csv(out/'primary_summary.csv', index=False)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for i, (name, *_rest) in enumerate(cases()):
        sub = primary_rows[(primary_rows['case']==name) & (primary_rows.n==max(sizes))]
        sub = sub[sub.rho0 == sub.rho1]
        for method in METHODS:
            q = sub[sub.method==method].sort_values('rho0')
            axes[i, 0].plot(q.rho0, q.strict_wrong_rate, 'o-', label=method)
            err = np.maximum(0, np.vstack([q.mean_loss-q.loss_low, q.loss_high-q.mean_loss]))
            axes[i, 1].errorbar(q.rho0, 1000*q.mean_loss, yerr=1000*err, fmt='o-', label=method)
        axes[i, 0].set(title=f'{name}, n={max(sizes)}', ylabel='Strict wrong-ranking rate', ylim=(-.03, 1.03))
        axes[i, 1].set(title=name, ylabel='Expected NB loss per 1000')
        for ax in axes[i]: ax.set_xlabel('Conditional latent Gaussian rho'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(out/'primary_sensitivity.png', dpi=180); plt.close(fig)
    (out/'README.txt').write_text(
        'Fixed-model evaluation, not model training. rho0/rho1 are latent conditional correlations.\n'
        'Population marginals, calibration and both population NB targets are invariant by construction.\n'
        'Quantile inversion is numerical; inspect quantile_audit.csv. Diagnostics are not calibration proofs.\n'
        'Paired changes use common random numbers versus (0,0). Intervals/MCSE measure simulation precision.\n'
        'Loss is max(true NB_A,true NB_B)-true NB of the selected model; ties choose A.\n'
        'Known-mean Beta is an oracle sensitivity comparator, not a deployable proposal.\n'
        'Loss per 1000 is NB-equivalent utility, not observed adverse events.\n'
        'Prior construction: Pfeiffer & Gail (2020), DOI 10.1002/bimj.201800240, Section 4.3.\n'
        'This Gaussian-copula grid does not cover all possible dependence structures.\n', encoding='utf-8')
    if source is not None:
        (out/'executed_script.py').write_bytes(source)
        (out/'script_sha256.txt').write_text(hashlib.sha256(source).hexdigest())
    archive = shutil.make_archive(str(out), 'zip', root_dir=out)
    print('\nDONE:', archive)
    print(primary_rows[['case','n','rho0','rho1','method','strict_wrong_rate','mean_loss']].to_string(index=False))
    return result

if __name__ == '__main__':
    run()
