"""Offline replay and bounded implementation audit. No model fitting or full simulation.
Run: python verify.py
Optional refitting is explicitly separate; see README.txt.
"""
from pathlib import Path
import sys, json, hashlib, platform, tempfile, zipfile, importlib.metadata
import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'analysis'))
import finite_generate as fg
import postprocess as pp
REPORT={}
TOL=1e-10

def close(a,b,label,tol=TOL):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError((label,'shape or finite check',a.shape,b.shape))
    e=float(np.max(abs(a-b))) if a.size else 0.
    if e>tol:raise AssertionError((label,e,tol))
    return e

def bnb(mu,var,t):
    if not (0<mu<1 and 0<var<mu*(1-mu)):raise ValueError('Invalid Beta moments')
    k=mu*(1-mu)/var-1
    return (mu*beta.sf(t,mu*k+1,(1-mu)*k)-t*beta.sf(t,mu*k,(1-mu)*k))/(1-t)

def direct(y,p,t,anchored=False):
    c=[np.mean((p[:,None]>=t)*(y[:,None]-t)/(1-t),axis=0),
       np.maximum(p[:,None]-t,0).mean(axis=0)/(1-t),bnb(p.mean(),p.var(),t)]
    if anchored:c.append(bnb(y.mean(),p.var(),t))
    return np.array(c)

def prepared(y,p,t):
    ix=np.argsort(p,kind='stable'); q=p[ix]
    return ix,q,y[ix],np.searchsorted(q,t,side='left')

def weighted(info,w,t):
    ix,p,y,cut=info;w=w[ix];n=w.sum()
    def tail(v):
        c=np.r_[0.,np.cumsum(w*v)];return c[-1]-c[cut]
    mu=np.sum(w*p)/n;v=np.sum(w*(p-mu)**2)/n
    return np.array([(tail(y)-t*tail(np.ones(len(p))))/(n*(1-t)),
                     (tail(p)-t*tail(np.ones(len(p))))/(n*(1-t)),bnb(mu,v,t)])

def clinical(name):
    folder=ROOT/'inputs'/name
    support=name=='support'
    pred=pd.read_csv(folder/('test_predictions.csv' if support else 'predictions.csv'))
    pair=pd.read_csv(folder/('paired_comparison.csv' if support else 'paired_curves.csv'))
    proto=json.loads((ROOT/'protocols'/(name+'.json')).read_text())
    t=np.array(proto['thresholds']); y=pred['outcome' if support else 'y'].to_numpy()
    names=['observed','proxy','beta']+(['prevalence_beta'] if support else [])
    statuses=['raw','cal'];arrays={c:pred[c].to_numpy() for c in ['LR_raw','LR_cal','HGB_raw','HGB_cal']}
    for p in arrays.values():
        if not np.isfinite(p).all() or not ((0<=p)&(p<=1)).all():raise AssertionError('Invalid predictions')
    membership=pd.read_csv(folder/'split_membership.csv')
    idcol='id' if support else 'row_id'
    if not membership[idcol].is_unique:raise AssertionError('Non-unique split IDs')
    if set(pred[idcol])!=set(membership.loc[membership.split=='test',idcol]):raise AssertionError('Test ID mismatch')
    point=np.array([direct(y,arrays['LR_'+s],t,support)-direct(y,arrays['HGB_'+s],t,support) for s in statuses])
    err=0.
    for si,s in enumerate(statuses):
        q=pair[pair.status==s].sort_values('threshold')
        close(q.threshold,t,name+' thresholds')
        for ei,e in enumerate(names):err=max(err,close(point[si,ei],q[e+'_delta' if support else e],name+' point '+e))
    n=len(y); B=proto['bootstrap_n'] if support else proto['bootstrap']
    seed=proto['bootstrap_seed'] if support else proto['seed']+1
    rng=np.random.default_rng(seed);boots=np.empty((B,2,len(names),len(t)))
    infos={k:prepared(y,p,t) for k,p in arrays.items()}
    for i in range(B):
        ix=rng.integers(0,n,size=n)
        if support:c={k:direct(y[ix],p[ix],t,True) for k,p in arrays.items()}
        else:
            w=np.bincount(ix,minlength=n);c={k:weighted(v,w,t) for k,v in infos.items()}
        boots[i]=[c['LR_'+s]-c['HGB_'+s] for s in statuses]
    lo,hi=np.quantile(boots,[.025,.975],axis=0);cierr=0.
    for si,s in enumerate(statuses):
        q=pair[pair.status==s].sort_values('threshold')
        for ei,e in enumerate(names):
            cierr=max(cierr,close(lo[si,ei],q[e+'_lo'],name+' low'),close(hi[si,ei],q[e+'_hi'],name+' high'))
    result=dict(n=n,events=int(y.sum()),point_max_error=err,bootstrap_replays=B,percentile_interval_max_error=cierr)
    if not support:
        saved=np.load(folder/'bootstrap_differences.npz')['differences']
        result['all_bootstrap_array_max_error']=close(boots,saved,'GUSTO saved bootstrap')
        radius=float(np.quantile(np.max(abs(boots-point),axis=(1,2,3)),.95))
        target=json.loads((folder/'metadata.json').read_text())['joint_band_radius']
        result['joint_band_radius']=radius
        result['joint_band_radius_error']=close(radius,target,'GUSTO radius')
        for si,s in enumerate(statuses):
            q=pair[pair.status==s].sort_values('threshold')
            for ei,e in enumerate(names):
                close(point[si,ei]-radius,q[e+'_sim_lo'],'GUSTO lower band')
                close(point[si,ei]+radius,q[e+'_sim_hi'],'GUSTO upper band')
    REPORT[name]=result
    print(name, result,flush=True)

def finite():
    folder=ROOT/'inputs/finite';d=pd.read_csv(folder/'replicates.csv');summary=pd.read_csv(folder/'summary.csv')
    pop=pd.read_csv(folder/'population.csv');quality=pd.read_csv(folder/'population_quality.csv')
    assert len(d)==10800 and not d.duplicated(['case','n','rep','threshold']).any()
    errs=[]
    for row in summary.itertuples():
        g=d[(d.case==row.case)&(d.n==row.n)&np.isclose(d.threshold,row.threshold)]
        assert len(g)==200
        for col in ['beta_wrong','observed_wrong','known_mu_beta_wrong','beta_wrong_ipd_ci_correct','observed_ci_covers','beta_tie','observed_tie']:
            if col+'_rate' in summary:
                v,lo,hi=fg.wilson(g[col])
                errs.append(close(v,getattr(row,col+'_rate'),'simulation rate'))
                if col+'_mc_lo' in summary:
                    errs.extend([close(lo,getattr(row,col+'_mc_lo'),'Wilson lower'),close(hi,getattr(row,col+'_mc_hi'),'Wilson upper')])
    sampleerrs=[];poperrs=[];quaderrs=[]
    for ci,case in enumerate(fg.CASES):
        pa,pb=(fg.matched(.2,.01),fg.matched(.2,.015)) if case=='beta_control' else (fg.make_mixture('A',.25 if case=='mixture_025' else .5),fg.make_mixture('B',.25 if case=='mixture_025' else .5))
        ta,tb=fg.population_nb(pa),fg.population_nb(pb)
        q=pop[pop.case==case].sort_values('threshold')
        for label,x in [('A',pa),('B',pb)]:
            m,v=fg.moments(x)
            poperrs.extend([close(fg.population_nb(x),q['true_'+label],'population true'),close(fg.population_nb(fg.matched(m,v)),q['beta_'+label],'population Beta')])
            aq=quality[(quality.case==case)&(quality.model==label)].iloc[0]
            poperrs.extend([close(fg.calibrated_auc(x),aq.auc,'AUC'),close(m*(1-m)-v,aq.brier,'Brier')])
            # Independent component-wise quadrature for the primary threshold, including narrow modes.
            aa,bb,ww=x;t=.2;integrals=[]
            for a,b in zip(aa,bb):
                mode=a/(a+b);sd=np.sqrt(a*b/((a+b)**2*(a+b+1)))
                breaks=sorted(set([t,1.]+[float(z) for z in mode+np.array([-8,-4,-2,0,2,4,8])*sd if t<z<1]))
                val=sum(quad(lambda s:(s-t)/(1-t)*beta.pdf(s,a,b),l,u,epsabs=1e-11,limit=200)[0] for l,u in zip(breaks[:-1],breaks[1:]))
                integrals.append(val)
            quaderrs.append(close(ww@np.array(integrals),fg.population_nb(x)[2],'independent integral',1e-8))
        for ni,n in enumerate(fg.SIZES):
            rng=np.random.default_rng(np.random.SeedSequence([fg.SEED,ci,ni]))
            # Only rep=0 in each fixed stratum: nine diagnostic replays, not a new experiment.
            y=rng.binomial(1,fg.MU,n);a=fg.sample_given_y(pa,y,rng);b=fg.sample_given_y(pb,y,rng)
            c=((a[:,None]>=fg.TS).astype(int)-(b[:,None]>=fg.TS).astype(int))*(y[:,None]-fg.TS)/(1-fg.TS)
            observed=c.mean(0);se=c.std(0,ddof=1)/np.sqrt(n)
            proxy=(np.maximum(a[:,None]-fg.TS,0)-np.maximum(b[:,None]-fg.TS,0)).mean(0)/(1-fg.TS)
            estimated=fg.beta_estimate(a)-fg.beta_estimate(b)
            z=d[(d.case==case)&(d.n==n)&(d.rep==0)].sort_values('threshold')
            for col,x in [('observed_delta',observed),('proxy_delta',proxy),('estimated_beta_delta',estimated),('observed_ci_low',observed-1.959963984540054*se),('observed_ci_high',observed+1.959963984540054*se)]:
                sampleerrs.append(close(x,z[col],'rep=0 '+col))
    REPORT['finite']=dict(rows=len(d),summary_max_error=max(errs),population_max_error=max(poperrs),independent_integration_max_error=max(quaderrs),diagnostic_replayed_datasets=9,diagnostic_replay_max_error=max(sampleerrs),full_1800_dataset_simulation_rerun=False)
    print('finite',REPORT['finite'],flush=True)

def postprocess():
    # Assemble the expected interface in a temporary file without modifying archived inputs.
    with tempfile.TemporaryDirectory() as td:
        zpath=Path(td)/'gusto.zip'
        with zipfile.ZipFile(zpath,'w') as z:
            for p in (ROOT/'inputs/gusto').iterdir():
                if p.is_file():z.write(p,p.name)
        pp.main(zpath)
    maxerr=0.;checked=0
    for name in ['error_decomposition.csv','bin_summary.csv','bin_details.csv']:
        a=pd.read_csv(ROOT/'verified_outputs/postprocess'/name);b=pd.read_csv(ROOT/'inputs/postprocess'/name)
        assert a.shape==b.shape and list(a)==list(b)
        for col in a:
            if pd.api.types.is_numeric_dtype(a[col]):maxerr=max(maxerr,close(a[col],b[col],name+'/'+col))
            else:assert a[col].equals(b[col]),(name,col)
        checked+=len(a)
    REPORT['postprocess']=dict(rows=checked,max_error=maxerr)

def source_semantics():
    # Direct finite-vector equality of the author's risk-sum expression and hinge expression.
    p=np.array([0.,.1,.2,.2,.3,1.]);t=np.array([.1,.2,.3,.8])
    flags=p[:,None]>=t
    authors=((flags*p[:,None]).sum(0)-(flags*(1-p[:,None])).sum(0)*t/(1-t))/len(p)
    hinge=np.maximum(p[:,None]-t,0).mean(0)/(1-t)
    e=close(authors,hinge,'author risk-sum semantics')
    REPORT['source_semantics']=dict(vector_identity_error=e,includes_exact_threshold_ties=True,native_stata_executed=False,scope='Standard single-model untruncated formula only; not generalized DCA, graphical defaults, or multi-model averaging')

def main():
    manifest=json.loads((ROOT/'audit/input_sha256.json').read_text())
    for p,sha in manifest.items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=sha:raise AssertionError('Input hash mismatch: '+p)
    REPORT['input_files_verified']=len(manifest)
    REPORT['environment']={'python':platform.python_version(),**{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy']}}
    source_semantics();clinical('support');clinical('gusto');finite();postprocess()
    REPORT['status']='PASS';REPORT['scope']='Offline outputs and bounded replays; no model refitting in this command'
    (ROOT/'verified_outputs/verification.json').write_text(json.dumps(REPORT,indent=2))
    print('PASS: see verified_outputs/verification.json')

if __name__=='__main__':main()
