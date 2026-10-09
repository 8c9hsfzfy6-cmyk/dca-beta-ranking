import json
import numpy as np
from scipy.integrate import quad
from scipy.special import expit, logit
from scipy.stats import norm, beta

# Zhu et al., Section 4.1.1: second normal parameter interpreted as SD.
# This agrees with the published Brier and NB table.
def quantities(m, sd, threshold=.3):
    def r(x):
        return expit(norm.logpdf(x, loc=m, scale=sd)-norm.logpdf(x))
    def e(g):
        # Integrate each unit interval separately so narrow regions are not missed.
        return sum(quad(lambda x: g(r(x))*.5*(norm.pdf(x)+norm.pdf(x,loc=m,scale=sd)),
                        lo, lo+1, epsabs=1e-12, epsrel=1e-11)[0]
                   for lo in range(-40,40))
    mean=e(lambda z:z)
    variance=e(lambda z:z*z)-mean**2
    A=(1-1/sd**2)/2
    B=m/sd**2
    C=-m*m/(2*sd*sd)-np.log(sd)-logit(threshold)
    roots=np.sort(np.roots([A,B,C]))
    cuts=[-np.inf,*roots,np.inf]
    nb=0.
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        x=hi-1 if np.isneginf(lo) else lo+1 if np.isposinf(hi) else (lo+hi)/2
        if A*x*x+B*x+C>0:
            p1=norm.cdf(hi,loc=m,scale=sd)-norm.cdf(lo,loc=m,scale=sd)
            p0=norm.cdf(hi)-norm.cdf(lo)
            nb+=.5*p1-.5*threshold/(1-threshold)*p0
    return dict(mean=mean,variance=variance,half_brier=e(lambda z:z*(1-z))/2,nb=nb)

def beta_nb(mu, variance, threshold):
    k=mu*(1-mu)/variance-1
    a,b=mu*k,(1-mu)*k
    return (mu*beta.sf(threshold,a+1,b)-threshold*beta.sf(threshold,a,b))/(1-threshold)

p1,p2=quantities(2,2),quantities(1,.5)
assert abs(p1['mean']-.5)<1e-10 and abs(p2['mean']-.5)<1e-10
assert abs(p1['variance']-p2['variance'])<1e-10
assert abs(p1['nb']-.3272393309614523)<1e-10
assert abs(p2['nb']-.3841596079739047)<1e-10
alpha=.01
v_h=.125 # Beta(.5,.5)
nb_h=beta_nb(.5,v_h,.3)
v_a=(1-alpha)*p1['variance']+alpha*v_h
nb_a=(1-alpha)*p1['nb']+alpha*nb_h
b_a=beta_nb(.5,v_a,.3)
b_b=beta_nb(.5,p2['variance'],.3)
assert nb_a<p2['nb'] and b_a>b_b
print(json.dumps(dict(prior_model_1=p1,prior_model_2=p2,
    beta_tie=beta_nb(.5,p1['variance'],.3),
    perturbation=dict(alpha=alpha,h_nb=nb_h,variance_a=v_a,variance_b=p2['variance'],
      true_nb_a=nb_a,true_nb_b=p2['nb'],beta_nb_a=b_a,beta_nb_b=b_b,
      true_difference=nb_a-p2['nb'],beta_difference=b_a-b_b)),indent=2))
