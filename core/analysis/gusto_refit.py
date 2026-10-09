"""Consolidated retained algorithm; source identity of original Kaggle cell is unverified."""
"""Fixed exploratory GUSTO geographic evaluation; CPU, no tuning.
Not a clinical treatment-effect study or a replication of the 1995 model.
"""
import sys, subprocess, importlib.util, io, json, hashlib, zipfile, platform
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen
import pyreadr
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.special import expit, logit
from scipy.stats import beta
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from threadpoolctl import threadpool_limits

SEED, BOOT = 2026100845, 500
TS = np.arange(1, 31) / 100
URL = 'https://raw.githubusercontent.com/resplab/predtools/master/data/gusto.rda'
SHA = 'e12bc58730894fa26f31b5b4ea963a878e855d7f2d53e47991cf8bb80b84d8e1'
BASE = Path(__file__).resolve().parents[1]
ROOT = BASE/'verified_outputs/gusto_refit'
PROTOCOL=json.loads((BASE/'protocols/gusto.json').read_text())
assert PROTOCOL['seed']==SEED and PROTOCOL['bootstrap']==BOOT
assert np.array_equal(PROTOCOL['thresholds'],TS)

def calfit(p, y):
    X = np.column_stack([np.ones(len(p)), logit(np.clip(p, 1e-8, 1-1e-8))])
    def fun(z):
        h = X @ z
        return np.mean(np.logaddexp(0, h)-y*h), X.T @ (expit(h)-y)/len(y)
    fit = minimize(fun, [0., 1.], jac=True, method='BFGS', options={'gtol': 1e-8})
    if not fit.success and np.max(np.abs(fit.jac)) > 1e-6:
        raise RuntimeError(str(fit.message))
    return fit.x

def prep(p, y):
    order = np.argsort(p, kind='stable')
    q = p[order]
    return order, q, y[order], np.searchsorted(q, TS, side='left')

def curves(info, weights):
    order, p, y, cut = info
    w = weights[order]
    n = w.sum()
    def tail(v):
        c = np.r_[0., np.cumsum(w*v)]
        return c[-1]-c[cut]
    count, events, risk = tail(np.ones(len(p))), tail(y), tail(p)
    observed = (events-TS*count)/(n*(1-TS))
    proxy = (risk-TS*count)/(n*(1-TS))
    mu = np.sum(w*p)/n
    var = np.sum(w*(p-mu)**2)/n
    if not (0 < var < mu*(1-mu)):
        raise ValueError('Invalid Beta moments; no silent clipping')
    k = mu*(1-mu)/var-1
    a, b = mu*k, (1-mu)*k
    approx = (mu*beta.sf(TS, a+1, b)-TS*beta.sf(TS, a, b))/(1-TS)
    return np.array([observed, proxy, approx])

def opposite(a, b):
    return ((a > 1e-8) & (b < -1e-8)) | ((a < -1e-8) & (b > 1e-8))

def main():
    source = BASE/'inputs/gusto/gusto.rda'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SHA, 'Source changed: stop and audit'
    df = pyreadr.read_r(str(source))['gusto'].reset_index(drop=True)
    assert df.shape == (40830, 29)
    assert df.day30.notna().all() and set(df.day30.unique()) == {0, 1}
    assert df.regl.notna().all()
    expected = {'Killip': {'I','II','III','IV'}, 'pmi': {'no','yes'},
                'miloc': {'Inferior','Other','Anterior'}}
    for col, levels in expected.items():
        assert df[col].notna().all() and set(df[col].unique()) == levels
    pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing': df.isna().sum()}).to_csv(ROOT/'data_audit.csv')
    y = df.day30.to_numpy(dtype=int)
    # Six predictors, manually encoded; no data-dependent feature selection.
    X = pd.DataFrame({'age': df.age, 'pulse': df.pulse.where(df.pulse > 0),
        'sbp': df.sysbp.where(df.sysbp > 0).clip(upper=100),
        'kill': (df.Killip != 'I').astype(int), 'pmi': (df.pmi == 'yes').astype(int),
        'miloc_other': (df.miloc == 'Other').astype(int),
        'miloc_anterior': (df.miloc == 'Anterior').astype(int)})
    assert np.isfinite(X.dropna().to_numpy()).all()
    X.isna().sum().to_csv(ROOT/'feature_missingness.csv')
    us = df.regl.isin(PROTOCOL['us_region_codes']).to_numpy()
    train = np.flatnonzero(~us)
    cal, test = train_test_split(np.flatnonzero(us), test_size=.5,
        random_state=SEED, stratify=y[us])
    membership = np.full(len(y), 'train', dtype=object)
    membership[cal], membership[test] = 'calibration', 'test'
    pd.DataFrame({'row_id': np.arange(len(y)), 'region': df.regl,
                  'split': membership}).to_csv(ROOT/'split_membership.csv', index=False)
    splits = pd.DataFrame([dict(split=s, n=len(i), events=int(y[i].sum()))
        for s,i in [('train',train), ('calibration',cal), ('test',test)]])
    splits.to_csv(ROOT/'splits.csv', index=False)
    models = {'LR': LogisticRegression(penalty=None, max_iter=3000, tol=1e-8),
        'HGB': HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
            learning_rate=.05, l2_regularization=1., min_samples_leaf=30,
            early_stopping=False, random_state=SEED)}
    predictions, calibration = {}, {}
    for name, model in models.items():
        pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), model)
        pipeline.fit(X.iloc[train], y[train])
        pc = pipeline.predict_proba(X.iloc[cal])[:,1]
        pt = pipeline.predict_proba(X.iloc[test])[:,1]
        z = calfit(pc, y[cal])
        calibration[name] = z.tolist()
        predictions[name+'_raw'] = pt
        predictions[name+'_cal'] = expit(z[0]+z[1]*logit(np.clip(pt,1e-8,1-1e-8)))
    (ROOT/'recalibration.json').write_text(json.dumps(calibration, indent=2))
    yt = y[test]
    pd.DataFrame({'row_id': test, 'y': yt, **predictions}).to_csv(ROOT/'predictions.csv', index=False)
    infos = {k: prep(p, yt) for k,p in predictions.items()}
    fitted = {k: curves(v, np.ones(len(test))) for k,v in infos.items()}
    metrics, calibration_bins, model_rows = [], [], []
    for name,p in predictions.items():
        obs, proxy, approx = fitted[name]
        z = calfit(p, yt)  # Diagnostic only: NEVER apply test-fitted coefficients.
        metrics.append(dict(model=name, auc=roc_auc_score(yt,p), brier=brier_score_loss(yt,p),
            mean_risk=p.mean(), prevalence=yt.mean(), test_cal_intercept=z[0], test_cal_slope=z[1],
            max_shape_error=np.max(abs(approx-proxy)), max_proxy_outcome_gap=np.max(abs(proxy-obs))))
        for j,t in enumerate(TS):
            model_rows.append(dict(model=name, threshold=t, observed=obs[j], proxy=proxy[j],
                beta=approx[j], shape_error=approx[j]-proxy[j],
                calibration_plus_sampling=proxy[j]-obs[j]))
        for g in np.array_split(np.argsort(p,kind='stable'),10):
            calibration_bins.append(dict(model=name,n=len(g),mean_risk=p[g].mean(),event_rate=yt[g].mean()))
    pd.DataFrame(metrics).to_csv(ROOT/'metrics.csv', index=False)
    pd.DataFrame(model_rows).to_csv(ROOT/'model_curves.csv', index=False)
    pd.DataFrame(calibration_bins).to_csv(ROOT/'calibration_bins.csv', index=False)
    statuses = ['raw','cal']
    point = np.array([fitted['LR_'+s]-fitted['HGB_'+s] for s in statuses])
    rng = np.random.default_rng(SEED+1)
    boots = np.empty((BOOT, 2, 3, len(TS)))
    for b in range(BOOT):
        w = np.bincount(rng.integers(0,len(test),len(test)), minlength=len(test))
        cc = {k: curves(v,w) for k,v in infos.items()}
        boots[b] = [cc['LR_'+s]-cc['HGB_'+s] for s in statuses]
    low, high = np.quantile(boots,[.025,.975],axis=0)
    # One common radius: joint across BOTH statuses, THREE estimands, ALL thresholds.
    radius = float(np.quantile(np.max(abs(boots-point),axis=(1,2,3)),.95))
    rows, summaries = [], []
    for si,status in enumerate(statuses):
        d = point[si]
        for j,t in enumerate(TS):
            row = dict(status=status, threshold=t)
            for e,name in enumerate(['observed','proxy','beta']):
                row.update({name:d[e,j], name+'_lo':low[si,e,j], name+'_hi':high[si,e,j],
                    name+'_sim_lo':d[e,j]-radius, name+'_sim_hi':d[e,j]+radius})
            row['shape_reversal'] = bool(opposite(d[1,j],d[2,j]))
            row['outcome_reversal'] = bool(opposite(d[0,j],d[2,j]))
            row['shape_reversal_joint'] = bool(opposite(d[1,j],d[2,j]) and min(abs(d[1,j]),abs(d[2,j])) > radius)
            row['outcome_reversal_joint'] = bool(opposite(d[0,j],d[2,j]) and min(abs(d[0,j]),abs(d[2,j])) > radius)
            rows.append(row)
        selected = rows[-len(TS):]
        summaries.append(dict(status=status, thresholds=len(TS),
            shape_reversals=sum(r['shape_reversal'] for r in selected),
            outcome_reversals=sum(r['outcome_reversal'] for r in selected),
            shape_reversals_joint=sum(r['shape_reversal_joint'] for r in selected),
            outcome_reversals_joint=sum(r['outcome_reversal_joint'] for r in selected)))
    pd.DataFrame(rows).to_csv(ROOT/'paired_curves.csv', index=False)
    pd.DataFrame(summaries).to_csv(ROOT/'summary.csv', index=False)
    np.savez_compressed(ROOT/'bootstrap_differences.npz',differences=boots)
    (ROOT/'metadata.json').write_text(json.dumps(dict(python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
        sklearn=sklearn.__version__, joint_band_radius=radius),indent=2))
    archive = ROOT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.iterdir()): z.write(p,p.name)
    print(splits.to_string(index=False))
    print(pd.DataFrame(metrics).round(6).to_string(index=False))
    print(pd.DataFrame(summaries).to_string(index=False))
    print('Joint band radius:',radius)
    print('Results ZIP:',archive)

if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    with threadpool_limits(limits=1): main()
