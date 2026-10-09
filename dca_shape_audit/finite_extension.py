"""Finite-sample extension. Import shape_path.py from same folder.
Only the small smoke test has been run during preparation; full run is optional.
"""
from pathlib import Path
from itertools import product
import json,time,sys
import numpy as np
import pandas as pd
import scipy
from scipy.special import betaincc
from shape_path import components,nb,ODDS_MULT
METHODS=['beta_estimated_mean','risk_only','outcome_based','beta_known_mean']

def draw_given_y(c,y,mu,rng):
    w,a,b=c;s=np.empty(len(y))
    for yy in [0,1]:
        ix=np.flatnonzero(y==yy)
        prob=w*(a/(a+b) if yy else b/(a+b))/(mu if yy else 1-mu)
        assert abs(prob.sum()-1)<1e-12
        z=rng.choice(len(w),len(ix),p=prob/prob.sum())
        s[ix]=rng.beta(a[z]+yy,b[z]+1-yy)
    return s

def beta_fit(s,t,known_mu=None):
    m=s.mean() if known_mu is None else known_mu;v=s.var(ddof=0)
    if not 0<v<m*(1-m):return np.full(len(t),np.nan)
    k=m*(1-m)/v-1;a=m*k;b=(1-m)*k
    return (m*betaincc(a+1,b,t)-t*betaincc(a,b,t))/(1-t)

def estimates(sa,sb,y,t,mu):
    risk=lambda s: np.maximum(s[:,None]-t,0).mean(axis=0)/(1-t)
    obs=lambda s: ((y[:,None]-t)*(s[:,None]>=t)).mean(axis=0)/(1-t)
    return np.column_stack([beta_fit(sa,t)-beta_fit(sb,t),risk(sa)-risk(sb),obs(sa)-obs(sb),beta_fit(sa,t,mu)-beta_fit(sb,t,mu)])

def scenarios():
    # Full shape cross, no selection on reversal. Duplicate epsilon=0 laws removed.
    out=[]
    for ratio in [1.1,1.5]:
        out.append(dict(mu=.2,f=.05,ratio=ratio,k=50.,sa=.5,sb=.5,eps=0.))
        for sa,sb,eps in product([.1,.5,.9],[.1,.5,.9],[.5,1.]):
            out.append(dict(mu=.2,f=.05,ratio=ratio,k=50.,sa=sa,sb=sb,eps=eps))
    return out

def wilson(k,n):
    if not n:return (np.nan,np.nan)
    p=k/n;z=1.959963984540054;d=1+z*z/n
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    c=(p+z*z/(2*n))/d
    return c-h,c+h

def run_finite(out,repetitions=500,sizes=(500,3000,10000),seed=2026100807,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cases=scenarios();ids=[0,1,2,17,18,19,36,37] if smoke else list(range(len(cases)))
    cfg=dict(code_version='shape-path-finite-1.0',repetitions=repetitions,sizes=list(sizes),seed=seed,case_ids=ids,methods=METHODS,conditional_dependence='independent given shared Y',mu=.2,f=.05,kappa=50.,odds_multipliers=ODDS_MULT,near_tie_tolerance=1e-10,decision_tie_rule='choose A',numpy=np.__version__,scipy=scipy.__version__,python=sys.version)
    manifest=out/'run_config.json'
    if manifest.exists():assert json.loads(manifest.read_text())==cfg,'Existing output has a different configuration; use a new output directory.'
    else:manifest.write_text(json.dumps(cfg,indent=2))
    allrows=[];start=time.perf_counter();done=0
    for sid in ids:
        q=cases[sid];mu=q['mu'];va=q['f']*mu*(1-mu);vb=va*q['ratio']
        ac=components(mu,va,q['k'],q['sa'],q['eps']);bc=components(mu,vb,q['k'],q['sb'],q['eps'])
        t=np.array(ODDS_MULT)*mu/(1-mu);t=t/(1+t);truth=nb(ac,t)-nb(bc,t)
        for n in sizes:
            f=out/f'case_{sid:03d}_n{n}.npz'
            if f.exists():
                with np.load(f) as old:
                    arr=old['estimates'];assert arr.shape==(repetitions,len(t),4)
                    assert np.allclose(old['truth'],truth,atol=1e-14,rtol=0)
            else:
                arr=np.empty((repetitions,len(t),4))
                for r in range(repetitions):
                    rng=np.random.default_rng(np.random.SeedSequence([seed,sid,n,r]))
                    y=rng.binomial(1,mu,n);sa=draw_given_y(ac,y,mu,rng);sb=draw_given_y(bc,y,mu,rng)
                    arr[r]=estimates(sa,sb,y,t,mu)
                temp=f.with_suffix('.tmp.npz');np.savez_compressed(temp,estimates=arr,truth=truth,thresholds=t)
                temp.replace(f)
            for j,tt in enumerate(t):
                for m,name in enumerate(METHODS):
                    est=arr[:,j,m];valid=np.isfinite(est);e=est[valid];count=len(e)
                    if count:
                        error=e-truth[j];pickA=e>=-1e-10
                        regret=np.where(pickA,max(0.,-truth[j]),max(0.,truth[j]))
                        ties=np.abs(e)<=1e-10
                        wrong=(e*truth[j]<0)&(~ties) if abs(truth[j])>1e-10 else np.zeros(count,bool)
                        lo,hi=wilson(int(wrong.sum()),count)
                        vals=dict(bias=float(error.mean()),bias_mcse=float(error.std(ddof=1)/np.sqrt(count)) if count>1 else np.nan,rmse=float(np.sqrt(np.mean(error**2))),strict_wrong_rate=float(wrong.mean()) if abs(truth[j])>1e-10 else np.nan,wrong_wilson_low=lo,wrong_wilson_high=hi,tie_rate=float(ties.mean()),mean_regret=float(regret.mean()),regret_mcse=float(regret.std(ddof=1)/np.sqrt(count)) if count>1 else np.nan)
                    else:vals={}
                    allrows.append(dict(scenario_id=sid,**q,n=n,threshold=float(tt),true_delta=float(truth[j]),method=name,valid_reps=count,failed_reps=int(repetitions-count),**vals))
            done+=1
            pd.DataFrame(allrows).to_csv(out/'finite_summary.csv',index=False)
            print(f'{done}/{len(ids)*len(sizes)} configurations saved; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    return pd.DataFrame(allrows)
if __name__=='__main__':
    # Change only RUN_FULL after reading the protocol. CPU is sufficient.
    RUN_FULL=False
    base=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
    run_finite(base/('DCA_shape_full' if RUN_FULL else 'DCA_shape_smoke'),repetitions=500 if RUN_FULL else 20,sizes=(500,3000,10000) if RUN_FULL else (200,),smoke=not RUN_FULL)
