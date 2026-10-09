from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import betaincc
from shape_path import components,nb,cdf
R=Path(__file__).resolve().parent;df=pd.read_csv(R.parent/'dca_round_final/results/population/population_grid.csv')
# Deterministic evenly spaced rows, plus an explicitly identified low-endpoint example.
selected=list(np.linspace(0,len(df)-1,31,dtype=int))
q=df[(df.mu==.2)&(df.kappa==50)&(df.vA_fraction==.05)&(df.variance_ratio==1.1)&(df.shapeA==.1)&(df.shapeB==.5)&(df.epsilon==.5)&(df.odds_multiplier==.5)].iloc[0]
selected.append(int(q.name));errors=[];auc_errors=[]
for idx in selected:
    p=df.iloc[idx]
    for label in ['A','B']:
        c=components(p.mu,p['v'+label],p.kappa,p['shape'+label],p.epsilon);w,a,b=c;t=p.threshold
        val=quad(lambda u:float(np.sum(w*betaincc(a,b,u))),t,1,epsabs=1e-12,epsrel=1e-11,limit=200)[0]/(1-t)
        errors.append(abs(val-nb(c,t)[0]))
        auc=.5+quad(lambda u:float(cdf(c,u)[0]*(1-cdf(c,u)[0])),0,1,epsabs=1e-11,epsrel=1e-10,limit=200)[0]/(2*p.mu*(1-p.mu))
        auc_errors.append(abs(auc-p['AUC_'+label]))
assert max(errors)<1e-9
report=dict(independent_nb_integrals=len(errors),max_nb_discrepancy=max(errors),max_auc_grid_discrepancy=max(auc_errors),illustrative_example=q.to_dict())
(R.parent/'dca_round_final/analysis/independent_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
