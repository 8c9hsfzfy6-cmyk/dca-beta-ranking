"""Small deterministic audit of bin information; no simulation or clinical data.

Run from DCA_GitHub: python application/verify_binning.py
Uses only NumPy and pandas. Binning bounds are classical Jensen/chord bounds.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent/'binning_results'
OUT.mkdir(parents=True,exist_ok=True)
TOL=2e-12


def hinge(s,t):
    return np.maximum(np.asarray(s)-t,0)/(1-t)


def summarize(scores,edges):
    # Half-open bins [l,u), except the final bin includes 1.
    labels=np.minimum(np.searchsorted(edges,scores,side='right')-1,len(edges)-2)
    assert np.all(labels>=0) and np.all(labels<len(edges)-1)
    rows=[]
    for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        values=scores[labels==k]
        rows.append(dict(lo=lo,hi=hi,count=len(values),
                         risk_sum=float(values.sum()),risk_square_sum=float(values@values)))
    return pd.DataFrame(rows)


def bounds(bins,n,t):
    lower=upper=cap=0.
    for row in bins.itertuples():
        if row.count==0:continue
        w,m=row.count/n,row.risk_sum/row.count
        lower+=w*float(hinge(m,t))
        upper+=w*((row.hi-m)*float(hinge(row.lo,t))+(m-row.lo)*float(hinge(row.hi,t)))/(row.hi-row.lo)
        if row.lo<t<row.hi:
            cap+=w*(t-row.lo)*(row.hi-t)/((row.hi-row.lo)*(1-t))
    return lower,upper,cap


def audit_grid():
    fixtures={
        'spread_a':np.array([.05,.15,.4,.8,.2,.45,.65,.95]),
        'spread_b':np.array([.1,.25,.35,.7,.3,.4,.6,.9]),
        'ties_endpoints':np.array([0,.2,.5,1,.2,.5,1,0]),
        'constant':np.full(8,.2)}
    thresholds=[.05,1/9,.1,.15,.2,.375,.5,.7,.95,.99]
    rows=[]
    for name,scores in fixtures.items():
        for t in thresholds:
            previous=None
            for count in [5,10,20,40]:
                edges=np.linspace(0,1,count+1)
                lower,upper,cap=bounds(summarize(scores,edges),len(scores),t)
                exact=float(hinge(scores,t).mean())
                assert lower-TOL<=exact<=upper+TOL
                assert upper-lower<=cap+TOL
                on_edge=bool(np.any(np.isclose(edges,t,rtol=0,atol=1e-13)))
                if on_edge:assert max(abs(lower-exact),abs(upper-exact))<TOL
                if previous is not None:
                    assert lower>=previous[0]-TOL and upper<=previous[1]+TOL
                previous=(lower,upper)
                rows.append(dict(fixture=name,threshold=t,bins=count,edge_aligned=on_edge,
                    empirical_risk_only=exact,lower=lower,upper=upper,width=upper-lower,
                    straddling_bin_width_cap=cap))
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'bin_grid.csv',index=False)
    return frame


def same_summary_examples():
    rows=[];summaries=[]
    examples=[
        ('interior_point','concentrated',np.array([.3,.3]),np.array([0,.2,.4,1]),.3),
        ('interior_point','spread',np.array([.21,.39]),np.array([0,.2,.4,1]),.3),
        ('edge_variance','concentrated',np.array([.1,.1,.6,.6]),np.array([0,.2,.4,.8,1]),.4),
        ('edge_variance','spread',np.array([.1,.1,.45,.75]),np.array([0,.2,.4,.8,1]),.4)]
    for group,label,scores,edges,t in examples:
        n=len(scores);bins=summarize(scores,edges);h=hinge(scores,t)
        lo,up,cap=bounds(bins,n,t)
        # Aligned bins plus within-bin sum of squared risks recover h^2 exactly.
        reconstructed_square=np.nan
        if group=='edge_variance':
            above=bins[bins.lo>=t]
            reconstructed_square=float(((above.risk_square_sum-2*t*above.risk_sum+t*t*above['count'])/(1-t)**2).sum())
            assert abs(reconstructed_square-float(h@h))<TOL
        rows.append(dict(example=group,configuration=label,n=n,threshold=t,
                         lower=lo,upper=up,empirical_risk_only=float(h.mean()),
                         sample_se=float(h.std(ddof=1)/np.sqrt(n)),
                         direct_contribution_square_sum=float(h@h),
                         recovered_square_sum=reconstructed_square))
        bins['example']=group;bins['configuration']=label;summaries.append(bins)
    result=pd.DataFrame(rows);allbins=pd.concat(summaries,ignore_index=True)
    for group in ['interior_point','edge_variance']:
        left=allbins[(allbins.example==group)&(allbins.configuration=='concentrated')]
        right=allbins[(allbins.example==group)&(allbins.configuration=='spread')]
        assert np.allclose(left[['lo','hi','count','risk_sum']],right[['lo','hi','count','risk_sum']],rtol=0,atol=TOL)
    interior=result[result.example=='interior_point']
    assert abs(interior.empirical_risk_only.iloc[0]-interior.empirical_risk_only.iloc[1])>.06
    variance=result[result.example=='edge_variance']
    assert abs(variance.empirical_risk_only.iloc[0]-variance.empirical_risk_only.iloc[1])<TOL
    assert abs(variance.sample_se.iloc[0]-variance.sample_se.iloc[1])>.02
    result.to_csv(OUT/'same_bin_summaries.csv',index=False)
    allbins.to_csv(OUT/'example_bin_inputs.csv',index=False)
    return result


def main():
    grid=audit_grid();examples=same_summary_examples()
    aligned=grid[grid.edge_aligned]
    report=dict(status='PASS',grid_cells=len(grid),edge_aligned_cells=len(aligned),
        nested_refinement_comparisons=4*10*3,
        edge_reconstruction_max_error=float(np.max(np.abs(aligned[['lower','upper']].to_numpy()-aligned.empirical_risk_only.to_numpy()[:,None]))),
        same_summary_example_pairs=2,
        scope='Deterministic algebra checks. No random simulation, patient data, coverage test, or new inferential theorem.')
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));print(examples.to_string(index=False))


if __name__=='__main__':main()
