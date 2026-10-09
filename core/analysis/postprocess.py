"""Post-hoc diagnostics only. No training or new bootstrap simulations.
Usage: python dca_gusto_postprocess.py PATH_TO_GUSTO_RESULTS.zip
"""
import sys, io, json, zipfile, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

def bounds(p, thresholds, k):
    # Inputs are counts and within-bin means; no outcomes are used.
    b = np.minimum((p*k).astype(int), k-1)
    n = np.bincount(b, minlength=k)
    sums = np.bincount(b, weights=p, minlength=k)
    means = np.divide(sums, n, out=np.zeros(k), where=n>0)
    w, lower, upper = n/len(p), np.arange(k)/k, (np.arange(k)+1)/k
    h = lambda x: np.maximum(x[:,None]-thresholds, 0)/(1-thresholds)
    L = np.sum(w[:,None]*h(means), axis=0)
    U = np.sum(w[:,None]*((upper-means)[:,None]*h(lower)
               +(means-lower)[:,None]*h(upper))*k, axis=0)
    return L, U

def main(source):
    out = Path(__file__).resolve().parents[1]/'verified_outputs/postprocess'
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(source) as z:
        predictions = pd.read_csv(z.open('predictions.csv'))
        paired = pd.read_csv(z.open('paired_curves.csv'))
        single = pd.read_csv(z.open('model_curves.csv'))
        boots = np.load(io.BytesIO(z.read('bootstrap_differences.npz')))['differences']
        source_sha = hashlib.sha256(z.read('gusto.rda')).hexdigest()
    errors = []
    for r in single.itertuples():
        p, y, t = predictions[r.model].to_numpy(), predictions.y.to_numpy(), r.threshold
        mu, var = p.mean(), p.var()
        k = mu*(1-mu)/var-1
        obs = np.mean((p>=t)*(y-t)/(1-t))
        proxy = np.maximum(p-t,0).mean()/(1-t)
        approx = (mu*beta.sf(t,mu*k+1,(1-mu)*k)-t*beta.sf(t,mu*k,(1-mu)*k))/(1-t)
        errors.extend(abs(np.array([obs,proxy,approx])-[r.observed,r.proxy,r.beta]))
    statuses = ['raw','cal']
    ts = np.arange(1,31)/100
    for s in statuses:
        assert np.allclose(paired.loc[paired.status==s,'threshold'],ts)
    point = np.stack([paired[paired.status==s][['observed','proxy','beta']].to_numpy().T for s in statuses])
    ep = np.stack([point[:,2]-point[:,1], point[:,1]-point[:,0], point[:,2]-point[:,0]],axis=1)
    eb = np.stack([boots[:,:,2]-boots[:,:,1], boots[:,:,1]-boots[:,:,0], boots[:,:,2]-boots[:,:,0]],axis=2)
    lo,hi = np.quantile(eb,[.025,.975],axis=0)
    radius = float(np.quantile(np.max(abs(eb-ep),axis=(1,2,3)),.95))
    rows = []
    for si,s in enumerate(statuses):
        for ei,label in enumerate(['shape','calibration_plus_sampling','total']):
            for j,t in enumerate(ts):
                rows.append(dict(status=s,threshold=t,component=label,estimate=ep[si,ei,j],
                    point_lo=lo[si,ei,j],point_hi=hi[si,ei,j],
                    joint_lo=ep[si,ei,j]-radius,joint_hi=ep[si,ei,j]+radius))
    pd.DataFrame(rows).to_csv(out/'error_decomposition.csv',index=False)
    summary, detailed = [], []
    for grid,t in [('original',ts),('shifted',np.arange(1,30)/100+.005)]:
        for s in statuses:
            a,b = [predictions[x+'_'+s].to_numpy() for x in ['LR','HGB']]
            exact = np.maximum(a[:,None]-t,0).mean(0)/(1-t)-np.maximum(b[:,None]-t,0).mean(0)/(1-t)
            for k in [5,10,20,40]:
                la,ua = bounds(a,t,k); lb,ub = bounds(b,t,k)
                L,U = la-ub,ua-lb
                certified = (L>1e-10)|(U < -1e-10)
                interior = ~np.isclose(t*k,np.round(t*k),rtol=0,atol=1e-10)
                contains = (exact>=L-1e-10)&(exact<=U+1e-10)
                assert contains.all()
                assert np.all(np.sign(exact[certified])==np.sign(L[certified]+U[certified]))
                summary.append(dict(grid=grid,status=s,bins=k,n_thresholds=len(t),
                    certified=int(certified.sum()),interior_n=int(interior.sum()),
                    interior_certified=int(certified[interior].sum()),
                    max_width=float(np.max(U-L)),all_empirical_targets_contained=bool(contains.all())))
                for j,threshold in enumerate(t):
                    detailed.append(dict(grid=grid,status=s,bins=k,threshold=threshold,
                        lower=L[j],upper=U[j],empirical_proxy_difference=exact[j],
                        certified=bool(certified[j]),interior=bool(interior[j])))
    pd.DataFrame(summary).to_csv(out/'bin_summary.csv',index=False)
    pd.DataFrame(detailed).to_csv(out/'bin_details.csv',index=False)
    metadata = dict(source_zip=str(source),source_data_sha256=source_sha,
        direct_recalculation_max_error=float(max(errors)),error_joint_radius=radius,
        scope='Exploratory post-hoc reuse of existing data and bootstrap. No new training.',
        bin_scope='Deterministic bounds for empirical score proxy, NOT population confidence intervals or clinical NB guarantees.',
        information_cost='K bin counts and K score means per model, versus 2 moments for Beta.',
        novelty='Jensen/chord bounds and bootstrap are classical tools; no new theorem claimed.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
    print(metadata)

if __name__ == '__main__':
    main(sys.argv[1])
