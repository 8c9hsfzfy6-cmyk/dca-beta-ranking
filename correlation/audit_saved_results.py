"""Audit saved estimates only; never generates subjects or runs simulations."""
from pathlib import Path
import json, hashlib, argparse
import numpy as np
import pandas as pd

def audit(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root/'run_config.json').read_text())
    sha = hashlib.sha256((root/'executed_script.py').read_bytes()).hexdigest()
    assert sha == cfg['source_sha256'] == (root/'script_sha256.txt').read_text().strip()
    assert cfg['reps'] == 500 and cfg['sizes'] == [500, 3000, 10000]
    summary = pd.read_csv(root/'summary.csv')
    paired = pd.read_csv(root/'paired_changes.csv')
    diagnostics = pd.read_csv(root/'diagnostics.csv')
    population = pd.read_csv(root/'population_truth.csv')
    tol = cfg['tolerance']; base = cfg['rho_pairs'].index([0., 0.])
    maxima = dict(summary=0., paired=0., diagnostics=0., population=0.)
    checked = 0
    def compare(frame, expected, key):
        assert len(frame) == 1
        row = frame.iloc[0]
        for col, value in expected.items():
            error = abs(float(row[col])-float(value))
            maxima[key] = max(maxima[key], error)
            assert error < 1e-12, (key, col, row[col], value)
    def metrics(e, truth):
        ties = np.abs(e) <= tol
        wrong = ((e*truth < 0) & ~ties).astype(float)
        loss = np.where(e >= -tol, max(0., -truth), max(0., truth))
        return wrong, ties, loss
    def se(x):return np.std(x, ddof=1)/np.sqrt(len(x))
    for case in cfg['cases']:
        name, thresholds = case['name'], np.array(case['thresholds'])
        for n in cfg['sizes']:
            with np.load(root/f'{name}_n{n}.npz') as z:
                est, diag, truth = z['estimates'], z['diagnostics'], z['truth']
                assert est.shape == (500,7,len(thresholds),4)
                assert diag.shape == (500,7,11)
                assert np.isfinite(est).all() and np.isfinite(diag).all()
                assert np.allclose(z['thresholds'], thresholds, atol=1e-15,rtol=0)
            fields=['corr_all','corr_y0','corr_y1','mean_a','mean_b','var_a','var_b',
                    'cal_resid_a','cal_resid_b','cal_score_resid_a','cal_score_resid_b']
            # Common random numbers leave A exactly unchanged across copulas.
            for col in [3,5,7,9]:
                assert np.array_equal(diag[:,:,col], np.repeat(diag[:,0,col,None],7,axis=1))
            for k,(r0,r1) in enumerate(cfg['rho_pairs']):
                mask=(summary['case']==name)&(summary.n==n)&(summary.rho0==r0)&(summary.rho1==r1)
                pmask=(paired['case']==name)&(paired.n==n)&(paired.rho0==r0)&(paired.rho1==r1)
                dmask=(diagnostics['case']==name)&(diagnostics.n==n)&(diagnostics.rho0==r0)&(diagnostics.rho1==r1)
                for di,field in enumerate(fields):
                    compare(diagnostics[dmask&diagnostics.diagnostic.eq(field)],
                            dict(mean=diag[:,k,di].mean(),mcse=se(diag[:,k,di])),'diagnostics')
                for j,t in enumerate(thresholds):
                    compare(population[population['case'].eq(name)&np.isclose(population.threshold,t,atol=1e-14,rtol=0)],dict(true_delta=truth[j]),'population')
                    for mi,method in enumerate(cfg['methods']):
                        e = est[:,k,j,mi]
                        wrong,ties,loss=metrics(e,truth[j])
                        rw,_,rl=metrics(est[:,base,j,mi],truth[j])
                        m=mask&summary.method.eq(method)&np.isclose(summary.threshold,t,atol=1e-14,rtol=0)
                        compare(summary[m],dict(strict_wrong_rate=wrong.mean(),tie_rate=ties.mean(),
                            mean_loss=loss.mean(),loss_mcse=se(loss),selection_error_rate=np.mean(loss>0),
                            bias=(e-truth[j]).mean(),bias_mcse=se(e-truth[j]),
                            rmse=np.sqrt(np.mean((e-truth[j])**2)),estimate_sd=e.std(ddof=1),
                            valid_reps=500,failed_reps=0),'summary')
                        pm=pmask&paired.method.eq(method)&np.isclose(paired.threshold,t,atol=1e-14,rtol=0)
                        compare(paired[pm],dict(wrong_change=(wrong-rw).mean(),wrong_change_mcse=se(wrong-rw),
                            loss_change=(loss-rl).mean(),loss_change_mcse=se(loss-rl),paired_reps=500),'paired')
                        checked+=1
    q=pd.read_csv(root/'quantile_audit.csv')
    assert len(q)==12 and q.max_cdf_error.max()<=1e-7
    assert checked == len(summary) == 1344
    primary=summary[summary.primary].copy()
    primary.to_csv(out/'verified_primary.csv',index=False)
    p=paired[paired.primary&(paired.n==10000)].copy()
    for m in ['wrong','loss']:
        p[m+'_change_low']=p[m+'_change']-1.95996398454*p[m+'_change_mcse']
        p[m+'_change_high']=p[m+'_change']+1.95996398454*p[m+'_change_mcse']
    p.to_csv(out/'paired_primary_changes.csv',index=False)
    d=diagnostics[(diagnostics.n==10000)&diagnostics.diagnostic.str.startswith('corr')]
    d.pivot(index=['case','rho0','rho1'],columns='diagnostic',values='mean').to_csv(out/'observed_correlations.csv')
    report=dict(status='PASS',scope='Recalculation from archived arrays only; no simulation replay',
                configurations=63,paired_datasets=31500,independent_base_repetitions=4500,
                checked_summary_rows=checked,checked_estimates=672000,all_finite=True,
                max_absolute_discrepancies=maxima,max_tested_inverse_cdf_error=float(q.max_cdf_error.max()),
                executed_script_sha256=sha,environment=cfg,
                inputs={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.iterdir()) if f.is_file()})
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['inputs','environment']},indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('output');a=p.parse_args()
    audit(a.results,a.output)
