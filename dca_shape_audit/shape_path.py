"""Moment-preserving shape sensitivity; population calculations only by default.
Dependencies: numpy, pandas, scipy. No data downloads, GPU or model fitting.
"""
from pathlib import Path
from itertools import product
import json, hashlib
import numpy as np
import pandas as pd
from scipy.special import betainc, betaincc
from numpy.polynomial.legendre import leggauss
MU=[.05,.2,.5]; VA_FRAC=[.05,.15,.4]; RATIOS=[1.1,1.5]
KAPPA=[50.,200.]; SHAPE=[.1,.5,.9]; EPS=[0.,.1,.25,.5,.75,1.]
ODDS_MULT=[.25,.5,1.,2.,4.]

def components(mu,v,kappa,shape,eps):
    vmax=mu*(1-mu);k=vmax/v-1
    vc=((kappa+1)*v-vmax)/kappa
    assert vc>0
    wlo=vc/(vc+mu*mu);whi=(1-mu)**2/((1-mu)**2+vc)
    w=wlo+shape*(whi-wlo)
    c=np.array([mu-np.sqrt(vc*(1-w)/w),mu+np.sqrt(vc*w/(1-w))])
    assert np.all((c>0)&(c<1))
    return np.r_[1-eps,eps*w,eps*(1-w)],np.r_[mu*k,kappa*c],np.r_[(1-mu)*k,kappa*(1-c)]

def moments(c):
    w,a,b=c;mu=np.sum(w*a/(a+b));second=np.sum(w*a*(a+1)/((a+b)*(a+b+1)))
    return mu,second-mu*mu

def nb(c,t):
    w,a,b=c;t=np.atleast_1d(t)[:,None]
    return ((w*(a/(a+b)*betaincc(a+1,b,t)-t*betaincc(a,b,t))).sum(axis=1)/(1-t[:,0]))

def cdf(c,t):
    w,a,b=c;return (w*betainc(a,b,np.atleast_1d(t)[:,None])).sum(axis=1)

def population(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];maxerr=0.
    x,z=leggauss(512);x=(x+1)/2;z=z/2;sid=0
    for mu,f,ratio,k,sa,sb in product(MU,VA_FRAC,RATIOS,KAPPA,SHAPE,SHAPE):
        va=f*mu*(1-mu);vb=va*ratio
        ts=np.array(ODDS_MULT)*mu/(1-mu);ts=ts/(1+ts)
        a0=components(mu,va,k,sa,0);b0=components(mu,vb,k,sb,0)
        ag=components(mu,va,k,sa,1);bg=components(mu,vb,k,sb,1)
        db=nb(a0,ts)-nb(b0,ts);dg=nb(ag,ts)-nb(bg,ts)
        for eps in EPS:
            ac=components(mu,va,k,sa,eps);bc=components(mu,vb,k,sb,eps)
            for c,v in [(ac,va),(bc,vb)]:
                m,vv=moments(c);maxerr=max(maxerr,abs(m-mu),abs(vv-v))
            delta=nb(ac,ts)-nb(bc,ts)
            maxerr=max(maxerr,float(np.max(np.abs(delta-((1-eps)*db+eps*dg)))))
            # AUC via E|S-S'| identity; quadrature is descriptive, checked separately below.
            auc=[];endpoint=[];ks=[]
            for c,c0 in [(ac,a0),(bc,b0)]:
                F=cdf(c,x);auc.append(.5+np.sum(z*F*(1-F))/(2*mu*(1-mu)))
                endpoint.append(float(cdf(c,.01)[0]+1-cdf(c,.99)[0]))
                ks.append(float(np.max(np.abs(F-cdf(c0,x)))))
            for j,t in enumerate(ts):
                ec=float(-db[j]/(dg[j]-db[j])) if dg[j]>1e-12 else np.nan
                rows.append(dict(scenario_id=sid,mu=mu,vA=va,vB=vb,vA_fraction=f,variance_ratio=ratio,kappa=k,shapeA=sa,shapeB=sb,epsilon=eps,threshold=float(t),odds_multiplier=ODDS_MULT[j],beta_delta=float(db[j]),true_delta=float(delta[j]),reversal=bool(db[j]<-1e-10 and delta[j]>1e-10),loss_selecting_beta=max(0.,float(delta[j])),critical_epsilon=ec,AUC_A=auc[0],AUC_B=auc[1],endpoint_mass_A=endpoint[0],endpoint_mass_B=endpoint[1],cdf_distance_A=ks[0],cdf_distance_B=ks[1]))
            sid+=1
    assert maxerr<1e-12
    df=pd.DataFrame(rows);df.to_csv(out/'population_grid.csv',index=False)
    # All prespecified cells retained; grouping rates refer only to this chosen grid.
    summary=df.groupby(['mu','epsilon'],as_index=False).agg(cells=('reversal','size'),reversals=('reversal','sum'),max_loss=('loss_selecting_beta','max'),mean_loss=('loss_selecting_beta','mean'))
    summary.to_csv(out/'population_summary.csv',index=False)
    eligible=df[(df.endpoint_mass_A<.01)&(df.endpoint_mass_B<.01)]
    rev=eligible[eligible.reversal]
    report=dict(scope='Exploratory grid; fractions are not clinical error prevalence; all scores exactly calibrated by common-outcome construction',scenarios=sid,threshold_cells=len(df),max_moment_or_linearity_error=maxerr,reversal_cells=int(df.reversal.sum()),low_endpoint_cells=len(eligible),low_endpoint_reversal_cells=len(rev),by_epsilon=df.groupby('epsilon').reversal.agg(['sum','size']).reset_index().to_dict('records'),largest_loss_low_endpoint=rev.nlargest(1,'loss_selecting_beta').to_dict('records'))
    (out/'population_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return df
if __name__=='__main__':population(Path(__file__).resolve().parent/'population_results')
