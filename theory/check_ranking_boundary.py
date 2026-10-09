"""Classical-moment sensitivity analysis; no model fitting or patient resampling.
Run from archive root: python audit/check_ranking_boundary.py
Requires numpy, scipy, matplotlib. Output: audit/boundary_results.json and figure3.png.
"""
from pathlib import Path
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from check_sharp_bounds import bounds, witnesses

def critical(mu,v):
    m=min(mu,1-mu)
    return np.sqrt(v)/2 if v<=m*m else m*v/(m*m+v)

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    errors=[]; witness_errors=[]; checked=0; classifications=0
    for mu in [.05,.2,.5,.8,.95]:
        vmax=mu*(1-mu)
        for va in vmax*np.linspace(.01,.99,99):
            g=critical(mu,va)
            errors.append(abs(bounds(mu,va,mu)[1]-g/(1-mu)))
            for vb in vmax*np.linspace(.01,.99,99):
                if vb<=va:continue
                a,b=bounds(mu,va,mu),bounds(mu,vb,mu)
                delta_hi=a[1]-b[0]
                errors.append(abs(delta_hi-(g-vb)/(1-mu)))
                if abs(vb-g)>1e-12:
                    assert (delta_hi<0)==(vb>g)
                    classifications+=1
                checked+=1
            # Boundary equality attained by independent extremal marginals under common Y.
            for v,j in [(va,1),(g,0)]:
                x,p=witnesses(mu,v,mu)[j]
                witness_errors.append(max(abs(p.sum()-1),abs(p@x-mu),abs(p@(x*x)-(mu*mu+v)),abs(p@np.maximum(x-mu,0)/(1-mu)-g/(1-mu))))
    assert max(errors)<1e-12 and max(witness_errors)<1e-12
    examples=[]
    for mu,va,vb in [(.2,.084501,.087001),(.2,.001,.15),(.5,.01,.04),(.5,.01,.06)]:
        a,b=bounds(mu,va,mu),bounds(mu,vb,mu)
        examples.append(dict(mu=mu,vA=va,vB=vb,critical_vB=critical(mu,va),contrast_interval=[float(a[0]-b[1]),float(a[1]-b[0])],B_guaranteed=bool(vb>critical(mu,va))))
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.8),sharex=True,sharey=True)
    z=np.linspace(0,1,1001)
    for ax,mu in zip(axes,[.05,.2,.5]):
        vmax=mu*(1-mu);g=np.array([critical(mu,zz*vmax)/vmax for zz in z])
        ax.fill_between(z,0,z,color='#eeeeee')
        ax.fill_between(z,z,g,color='#f4d5a6')
        ax.fill_between(z,g,1,color='#b9d9e8')
        ax.plot(z,g,color='#173c50',lw=1.6)
        ax.plot(z,z,color='#999999',lw=.7)
        ax.set(xlim=(0,1),ylim=(0,1),title=rf'$\mu=t={mu:.2f}$',xlabel=r'$v_A/[\mu(1-\mu)]$')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel(r'$v_B/[\mu(1-\mu)]$')
    fig.legend(handles=[Patch(facecolor='#b9d9e8',label='B strictly better for all feasible distributions'),Patch(facecolor='#f4d5a6',label='Both strict rankings feasible'),Patch(facecolor='#eeeeee',label=r'Outside displayed comparison ($v_B\leq v_A$)')],loc='lower center',ncol=1,fontsize=9,frameon=False)
    fig.subplots_adjust(bottom=.31,wspace=.17,top=.88,left=.07,right=.99)
    fig.savefig(out/'figure3.png',dpi=220)
    result=dict(scope='Known population moments; exact calibration; t=mu; classical corollary, not new theory or a clinical experiment',pairs_checked=checked,strict_classifications_checked=classifications,max_bound_identity_error=max(errors),max_boundary_witness_error=max(witness_errors),examples=examples)
    (out/'boundary_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':run(HERE)
