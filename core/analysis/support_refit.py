"""Consolidated reconstruction, not authenticated historical inline source. Explicit opt-in refit."""
"""Small fixed-split SUPPORT2 pilot. CPU only; no tuning or seed search.
Run alongside pilot_protocol.json. No claim of external clinical validation.
"""
from pathlib import Path
import hashlib,json,platform,urllib.request
import numpy as np
import pandas as pd
import scipy,sklearn
from scipy.special import expit,logit
from scipy.optimize import minimize
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,brier_score_loss
from threadpoolctl import threadpool_limits

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'verified_outputs/support_refit'
P=json.loads((BASE/'protocols/support.json').read_text())
# Archived final protocol is flat; retained computational body used nested keys.
if 'split' in P and P['split']['seed'] != P['seed']:
    raise ValueError('Conflicting split seeds')
if 'bootstrap' in P and (P['bootstrap']['seed'],P['bootstrap']['n']) != (P['bootstrap_seed'],P['bootstrap_n']):
    raise ValueError('Conflicting bootstrap settings')
P['split']={'seed':P['seed']}
P['bootstrap']={'seed':P['bootstrap_seed'],'n':P['bootstrap_n']}
TS=np.array(P['thresholds']); TOL=1e-8

def calfit(p,y):
    x=logit(np.clip(p,1e-8,1-1e-8)); X=np.column_stack([np.ones(len(x)),x])
    def f(z):
        eta=X@z
        return float(np.mean(np.logaddexp(0,eta)-y*eta)),X.T@(expit(eta)-y)/len(y)
    opt=minimize(f,[0.,1.],jac=True,method='BFGS',options={'gtol':1e-8})
    if not opt.success and np.max(abs(opt.jac))>1e-6:
        raise RuntimeError(str(opt.message))
    return opt.x

def beta_nb(mu,var):
    if not (0<mu<1 and 0<var<mu*(1-mu)):
        raise ValueError('Invalid moments; no silent clipping')
    k=mu*(1-mu)/var-1; a=mu*k; b=(1-mu)*k
    return (mu*beta.sf(TS,a+1,b)-TS*beta.sf(TS,a,b))/(1-TS)

def curves(y,p):
    flag=p[:,None]>=TS
    obs=np.mean(flag*(y[:,None]-TS)/(1-TS),axis=0)
    proxy=np.mean(np.maximum(p[:,None]-TS,0)/(1-TS),axis=0)
    approx=beta_nb(p.mean(),p.var(ddof=0))
    anchored=beta_nb(y.mean(),p.var(ddof=0))
    # Residual = calibration discrepancy plus outcome sampling noise, NOT pure calibration error.
    return dict(observed=obs,proxy=proxy,beta=approx,prevalence_beta=anchored)

def rev(a,b):
    return ((a>TOL)&(b<-TOL))|((a<-TOL)&(b>TOL))

def main():
    src=BASE/'inputs/support/support2_source.csv'
    if not src.exists():
        raise FileNotFoundError('Use the archived source data; no automatic download')
    digest=hashlib.sha256(src.read_bytes()).hexdigest()
    assert digest==P['source_sha256'], 'Dataset changed: audit before rerunning'
    df=pd.read_csv(src); assert len(df)==9105 and df.id.is_unique
    assert df.hospdead.notna().all() and set(df.hospdead)=={0,1}
    num=P['numeric'];cat=P['categorical'];features=num+cat
    y=df.hospdead.to_numpy(dtype=int);idx=np.arange(len(df));seed=P['split']['seed']
    tr,rem=train_test_split(idx,test_size=.5,random_state=seed,stratify=y)
    ca,te=train_test_split(rem,test_size=.5,random_state=seed+1,stratify=y[rem])
    assert not(set(tr)&set(ca) or set(tr)&set(te) or set(ca)&set(te))
    pd.DataFrame([dict(split=s,n=len(i),events=int(y[i].sum()))
                  for s,i in [('train',tr),('calibration',ca),('test',te)]]).to_csv(ROOT/'splits.csv',index=False)
    membership=np.full(len(df),'',dtype=object)
    for s,i in [('train',tr),('calibration',ca),('test',te)]:membership[i]=s
    pd.DataFrame({'id':df.id,'split':membership}).to_csv(ROOT/'split_membership.csv',index=False)
    df[features].isna().sum().rename('missing_count').to_csv(ROOT/'feature_missingness.csv')
    models={'LR':LogisticRegression(**P['models']['LR'],random_state=seed),
            'HGB':HistGradientBoostingClassifier(**P['models']['HGB'],random_state=seed)}
    preds={};fits={};metrics=[];cs={};curve_rows=[]
    for name,model in models.items():
        pre=ColumnTransformer([
            ('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),num),
            ('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),
                                 OneHotEncoder(handle_unknown='ignore',sparse_output=False)),cat)])
        pipe=make_pipeline(pre,model)
        pipe.fit(df.iloc[tr][features],y[tr])
        pc=pipe.predict_proba(df.iloc[ca][features])[:,1]
        pt=pipe.predict_proba(df.iloc[te][features])[:,1]
        z=calfit(pc,y[ca]);fits[name]=z.tolist()
        for status,p in [('raw',pt),('cal',expit(z[0]+z[1]*logit(np.clip(pt,1e-8,1-1e-8))))]:
            key=name+'_'+status;preds[key]=p; c=curves(y[te],p);cs[key]=c
            testz=calfit(p,y[te]) # descriptive diagnostic, NEVER applied to predictions
            metrics.append(dict(model=key,auc=roc_auc_score(y[te],p),brier=brier_score_loss(y[te],p),
                                mean_risk=p.mean(),prevalence=y[te].mean(),
                                calibration_intercept=testz[0],calibration_slope=testz[1],
                                max_shape_error=np.max(abs(c['beta']-c['proxy'])),
                                mean_shape_error=np.mean(abs(c['beta']-c['proxy'])),
                                max_proxy_outcome_gap=np.max(abs(c['proxy']-c['observed']))))
            for j,t in enumerate(TS):
                d={v:float(a[j]) for v,a in c.items()}
                d.update(model=key,threshold=t,shape_error=d['beta']-d['proxy'],
                         calibration_plus_sampling=d['proxy']-d['observed'],
                         anchoring_shift=d['prevalence_beta']-d['beta'])
                assert abs(d['beta']-d['observed']-d['shape_error']-d['calibration_plus_sampling'])<1e-12
                curve_rows.append(d)
    pd.DataFrame(metrics).to_csv(ROOT/'pilot_metrics.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(ROOT/'pilot_curves.csv',index=False)
    pd.DataFrame(dict(id=df.id.to_numpy()[te],outcome=y[te],**preds)).to_csv(ROOT/'test_predictions.csv',index=False)
    rng=np.random.default_rng(P['bootstrap']['seed']); B=P['bootstrap']['n'];boots={}
    for status in ['raw','cal']:
        boots[status]={key:np.empty((B,len(TS))) for key in ['observed','proxy','beta','prevalence_beta']}
    for b in range(B):
        ii=rng.integers(len(te),size=len(te));yy=y[te][ii]
        for status in ['raw','cal']:
            A=curves(yy,preds['LR_'+status][ii]);C=curves(yy,preds['HGB_'+status][ii])
            for key in A: boots[status][key][b]=A[key]-C[key]
    pairs=[];summ=[]
    for status in ['raw','cal']:
        A=cs['LR_'+status];C=cs['HGB_'+status]
        delta={key:A[key]-C[key] for key in A}
        shape_rev=rev(delta['beta'],delta['proxy']);out_rev=rev(delta['beta'],delta['observed'])
        # Empirical regret from selecting the higher Beta NB of these two models.
        regret=np.where(delta['beta']>TOL,np.maximum(-delta['observed'],0),
                        np.where(delta['beta']<-TOL,np.maximum(delta['observed'],0),0))
        proxy_regret=np.where(delta['beta']>TOL,np.maximum(-delta['proxy'],0),
                              np.where(delta['beta']<-TOL,np.maximum(delta['proxy'],0),0))
        for j,t in enumerate(TS):
            row=dict(status=status,threshold=t,shape_reversal=bool(shape_rev[j]),
                     outcome_reversal=bool(out_rev[j]),empirical_regret=regret[j],proxy_regret=proxy_regret[j])
            for key,d in delta.items():
                lo,hi=np.quantile(boots[status][key][:,j],[.025,.975])
                row.update({key+'_delta':d[j],key+'_lo':lo,key+'_hi':hi})
            # Exploratory sign-stability; pointwise, conditional on fitted models.
            row['beta_tie']=bool(abs(row['beta_delta'])<=TOL)
            row['opposite_pointwise_signs']=(row['beta_lo']>0 and row['observed_hi']<0) or (row['beta_hi']<0 and row['observed_lo']>0)
            pairs.append(row)
        summ.append(dict(status=status,n_thresholds=len(TS),shape_reversals=int(shape_rev.sum()),
                         outcome_reversals=int(out_rev.sum()),max_empirical_regret=regret.max(),
                         max_proxy_regret=proxy_regret.max(),
                         max_gap_at_shape_reversal=float(np.max(abs(delta['proxy'][shape_rev]))) if shape_rev.any() else 0,
                         opposite_pointwise_signs=sum(r['opposite_pointwise_signs'] for r in pairs if r['status']==status)))
    pd.DataFrame(pairs).to_csv(ROOT/'paired_comparison.csv',index=False)
    pd.DataFrame(summ).to_csv(ROOT/'pilot_summary.csv',index=False)
    (ROOT/'pilot_metadata.json').write_text(json.dumps(dict(source_sha256=digest,
       protocol_sha256=hashlib.sha256((ROOT/'pilot_protocol.json').read_bytes()).hexdigest(),
       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
       sklearn=sklearn.__version__,calibration_fits=fits),indent=2))
    print(pd.DataFrame(metrics).to_string(index=False));print(pd.DataFrame(summ).to_string(index=False))

if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'pilot_protocol.json').write_text(json.dumps(P,indent=2))
    with threadpool_limits(limits=2):main()
