from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'dca_shape_audit'))
from shape_path import components,nb
from finite_extension import scenarios,estimates,draw_given_y,wilson
D=R/'results';out=R/'analysis';out.mkdir(exist_ok=True)
cfg=json.loads((D/'finite/run_config.json').read_text());df=pd.read_csv(D/'finite/finite_summary.csv');pg=pd.read_csv(D/'population/population_grid.csv')
assert cfg['repetitions']==500 and cfg['case_ids']==list(range(38)) and cfg['sizes']==[500,3000,10000]
assert len(df)==38*3*5*4 and len(list((D/'finite').glob('case_*.npz')))==114
assert not df.duplicated(['scenario_id','n','threshold','method']).any()
err={}; replay=0.; truths=0.;narrays=0; derived=[]
for sid,q in enumerate(scenarios()):
 mu=q['mu'];va=q['f']*mu*(1-mu);vb=va*q['ratio'];ac=components(mu,va,q['k'],q['sa'],q['eps']);bc=components(mu,vb,q['k'],q['sb'],q['eps'])
 for n in cfg['sizes']:
  raw=np.load(D/f'finite/case_{sid:03d}_n{n}.npz');a=raw['estimates'];t=raw['thresholds'];truth=raw['truth'];assert a.shape==(500,5,4) and np.isfinite(a).all();narrays+=a.size
  truths=max(truths,float(np.max(abs(truth-(nb(ac,t)-nb(bc,t))))))
  rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],sid,n,0]));y=rng.binomial(1,mu,n);sa=draw_given_y(ac,y,mu,rng);sb=draw_given_y(bc,y,mu,rng);e=estimates(sa,sb,y,t,mu);replay=max(replay,float(np.max(abs(e-a[0]))))
  for j in range(5):
   for m,name in enumerate(cfg['methods']):
    e=a[:,j,m];er=e-truth[j];ties=abs(e)<=1e-10;wrong=(e*truth[j]<0)&~ties;reg=np.where(e>=-1e-10,max(0.,-truth[j]),max(0.,truth[j]));lo,hi=wilson(int(wrong.sum()),500)
    vals=dict(bias=er.mean(),bias_mcse=er.std(ddof=1)/np.sqrt(500),rmse=np.sqrt(np.mean(er**2)),strict_wrong_rate=wrong.mean() if abs(truth[j])>1e-10 else np.nan,wrong_wilson_low=lo,wrong_wilson_high=hi,tie_rate=ties.mean(),mean_regret=reg.mean(),regret_mcse=reg.std(ddof=1)/np.sqrt(500))
    row=df[(df.scenario_id==sid)&(df.n==n)&(df.method==name)].iloc[j];assert abs(row.threshold-t[j])<1e-14
    for k,v in vals.items():
     if np.isfinite(v):err[k]=max(err.get(k,0),abs(v-row[k]))
    derived.append(dict(scenario_id=sid,n=n,threshold=t[j],method=name,decision_error_rate=float((reg>0).mean()),**vals))
assert max(err.values())<1e-12 and truths<1e-12 and replay<1e-12
# Same scenarios and thresholds, attach exact Beta contrast and diagnostics.
def pgrow(r):
 a=pg[(pg.mu==r.mu)&(pg.vA_fraction==r.f)&(pg.variance_ratio==r.ratio)&(pg.kappa==r.k)&(pg.shapeA==r.sa)&(pg.shapeB==r.sb)&(pg.epsilon==r.eps)&(abs(pg.threshold-r.threshold)<1e-12)]
 assert len(a)==1;return a.iloc[0]
extra=[]
for _,r in df.iterrows():
 p=pgrow(r);extra.append({k:p[k] for k in ['beta_delta','endpoint_mass_A','endpoint_mass_B','reversal','AUC_A','AUC_B']})
df=pd.concat([df.reset_index(drop=True),pd.DataFrame(extra)],axis=1)
der=pd.DataFrame(derived); df['decision_error_rate']=df['mean_regret']/df['true_delta'].abs()
assert np.allclose(df['decision_error_rate'],der['decision_error_rate'],atol=1e-9)
df.to_csv(out/'verified_results.csv',index=False)
# Each row here is one distribution-threshold setting at one sample size, not independent clinical data.
agg=df.groupby(['n','method','reversal'],as_index=False).agg(cells=('scenario_id','size'),mean_wrong=('strict_wrong_rate','mean'),median_wrong=('strict_wrong_rate','median'),min_wrong=('strict_wrong_rate','min'),max_wrong=('strict_wrong_rate','max'),mean_regret=('mean_regret','mean'),max_regret=('mean_regret','max'))
agg.to_csv(out/'descriptive_grid_summary.csv',index=False)
example=df[(df.ratio==1.1)&(df.sa==.1)&(df.sb==.5)&(df.eps==.5)&(abs(df.threshold-1/9)<1e-12)];example.to_csv(out/'illustrative_case.csv',index=False)
controls=df[df.eps==0];controls.to_csv(out/'beta_controls.csv',index=False)
report=dict(configurations=114,repetitions_per_configuration=500,replicates=57000,raw_scalar_estimates=narrays,summary_rows=len(df),failed_estimates=int(df.failed_reps.sum()),max_summary_errors=err,max_truth_error=truths,max_first_replicate_replay_error=replay,first_replicates_replayed=114,sha256_input='e6b3fa31e9c034d14035a104cd9e37c5eddbf0bef1a82748282765cf5bcfa52a')
(out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print('EXAMPLE');print(example[['scenario_id','n','method','true_delta','strict_wrong_rate','wrong_wilson_low','wrong_wilson_high','mean_regret','rmse']].to_string(index=False));print('AGGREGATES');print(agg.to_string(index=False));print('CONTROLS MEAN ERRORS');print(controls.groupby(['n','method']).strict_wrong_rate.agg(['mean','max']).to_string());print('REV CELLS',df[(df.n==10000)&(df.method=='beta_estimated_mean')].reversal.value_counts().to_dict())
