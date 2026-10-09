"""Deterministic audit, not a sampling experiment. Python + numpy + scipy."""
import json
import numpy as np
from scipy.optimize import linprog
from scipy.special import betaincc

def bounds(mu, v, t):
    assert 0 < mu < 1 and 0 < v < mu*(1-mu) and 0 < t < 1
    q = mu*mu+v
    r, ell = q/mu, (mu-q)/(1-mu)
    lo = max(0., mu-t, q-mu*t)
    if t <= r/2:
        hi = mu-t*mu*mu/q
    elif t >= (1+ell)/2:
        hi = (1-t)*v/((1-mu)**2+v)
    else:
        hi = (mu-t+np.sqrt((mu-t)**2+v))/2
    return np.array([lo, hi])/(1-t)

def witnesses(mu, v, t):
    q = mu*mu+v
    r, ell = q/mu, (mu-q)/(1-mu)
    low_pair = (np.array([ell,1.]), np.array([(1-mu)/(1-ell),(mu-ell)/(1-ell)]))
    high_pair = (np.array([0.,r]), np.array([1-mu/r,mu/r]))
    if t <= ell:
        lower = low_pair
    elif t >= r:
        lower = high_pair
    else:
        p1 = (q-mu*t)/(1-t)
        pt = (mu-q)/(t*(1-t))
        p0 = (q-mu+(1-mu)*t)/t
        lower = (np.array([0.,t,1.]), np.array([p0,pt,p1]))
    if t <= r/2:
        upper = high_pair
    elif t >= (1+ell)/2:
        upper = low_pair
    else:
        d = np.sqrt((mu-t)**2+v)
        x = np.array([t-d,t+d])
        p = (mu-x[0])/(2*d)
        upper = (x, np.array([1-p,p]))
    return lower, upper

def beta_nb(mu,v,t):
    k = mu*(1-mu)/v-1
    a,b = mu*k,(1-mu)*k
    return (mu*betaincc(a+1,b,t)-t*betaincc(a,b,t))/(1-t)

def run_checks():
    maxerr = 0.
    count = 0
    lp_count = 0
    for mu in [.05,.2,.5,.8,.95]:
        for frac in [.01,.1,.5,.9,.99]:
            v = frac*mu*(1-mu)
            q = mu*mu+v
            r,ell = q/mu,(mu-q)/(1-mu)
            ts = sorted(set([.01,.1,.2,.5,.8,.9,.99,ell,r,r/2,(1+ell)/2]))
            for t in ts:
                exact = bounds(mu,v,t)
                ws = witnesses(mu,v,t)
                for j,(x,p) in enumerate(ws):
                    err = max(abs(p.sum()-1),abs(p@x-mu),abs(p@(x*x)-q),
                              abs(p@np.maximum(x-t,0)/(1-t)-exact[j]))
                    assert min(x) >= -1e-12 and max(x) <= 1+1e-12
                    assert min(p) >= -1e-12 and err < 2e-11
                    maxerr = max(maxerr,err)
                # LP is an independent optimization check on a finite support
                # containing both analytic witnesses; sharpness follows from proof.
                if t == .2 or t == .8:
                    x = np.unique(np.clip(np.r_[np.linspace(0,1,301),ws[0][0],ws[1][0]],0,1))
                    h = np.maximum(x-t,0)/(1-t)
                    mat = np.array([np.ones(len(x)),x,x*x])
                    for sign,j in [(1,0),(-1,1)]:
                        sol = linprog(sign*h,A_eq=mat,b_eq=[1,mu,q],bounds=(0,None),method='highs')
                        assert sol.success, sol.message
                        assert abs(sign*sol.fun-exact[j]) < 2e-7
                        lp_count += 1
                count += 1
    examples = []
    for name,mu,va,vb,t in [
        ('existing_lambda_half',.2,.084501,.087001,.2),
        ('certifiable_control',.2,.001,.15,.2),
    ]:
        ba,bb = bounds(mu,va,t),bounds(mu,vb,t)
        examples.append(dict(name=name,mu=mu,vA=va,vB=vb,t=t,
            nb_range_A=ba.tolist(),nb_range_B=bb.tolist(),
            difference_A_minus_B=[ba[0]-bb[1],ba[1]-bb[0]],
            beta_A=beta_nb(mu,va,t),beta_B=beta_nb(mu,vb,t),
            worst_loss_selecting_B=max(0.,ba[1]-bb[0])))
    return dict(moment_threshold_cases=count,finite_support_LP_checks=lp_count,
                max_witness_error=maxerr,examples=examples)

if __name__ == '__main__':
    print(json.dumps(run_checks(),indent=2))
