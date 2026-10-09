"""Small deterministic algebra audit; no MATLAB execution or simulation study.

Translations of the operations in NBinfRaloneempirical.m,
NBinfRYempirical.m, Table2code.m, Table4code.m, Figure2code.m.
The original source files are identified by hashes in the accompanying report.
"""
import json
from pathlib import Path
import numpy as np

def risk_influence(r, t):
    c = t / (1 - t)
    z = r > t
    return (1+c)*(r*z - np.mean(r*z))-c*(z-np.mean(z))

def outcome_influence(r, y, t):
    tp, fp = (r > t)*(y == 1), (r > t)*(y == 0)
    return (tp-tp.mean())-t/(1-t)*(fp-fp.mean())

def main():
    a = np.array([0, .1, .2, .2, .4, .8, 1.])
    b = np.array([.05, .3, .1, .2, .5, .7, .95])
    y = np.array([0, 1, 0, 1, 0, 1, 1])
    rows = []
    for t in [.1, .2, .4, .9]:
        c = t/(1-t)
        ha, hb = np.maximum(a-t, 0)/(1-t), np.maximum(b-t, 0)/(1-t)
        qa, qb = (y-t)*(a>t)/(1-t), (y-t)*(b>t)/(1-t)
        code_nb = (1+c)*np.mean(a*(a>t))-c*np.mean(a>t)
        ra, rb = risk_influence(a,t), risk_influence(b,t)
        oa, ob = outcome_influence(a,y,t), outcome_influence(b,y,t)
        checks = [abs(code_nb-ha.mean()), np.max(abs(ra-(ha-ha.mean()))),
                  np.max(abs(oa-(qa-qa.mean()))),
                  abs(np.mean((ra-rb)**2)-np.var(ha-hb)),
                  abs(np.mean((oa-ob)**2)-np.var(qa-qb))]
        assert max(checks) < 1e-13
        # >= differs for outcome data at a threshold atom, but not for a hinge.
        q_ge = (y-t)*(a>=t)/(1-t)
        tie_term = np.mean((y-t)*(a==t)/(1-t))
        assert abs((q_ge.mean()-qa.mean())-tie_term)<1e-13
        rows.append(dict(t=t,max_algebra_error=float(max(checks)),
                         outcome_ge_minus_gt=float(q_ge.mean()-qa.mean())))
    # Literal appendix section 3, with joint lower-tail probabilities as defined.
    t=.2; c=t/(1-t)
    # Cells: (R>t,Y=1), (R>t,Y=0), (R<=t,Y=1), (R<=t,Y=0).
    probs=np.array([.15,.25,.05,.55]); z=np.array([1,-c,0,0])
    p1,p0,f1,f0=probs
    true_var=float(np.dot(probs,z*z)-np.dot(probs,z)**2)
    literal_var=float(f1*(1-f1)+c*c*f0*(1-f0)-2*c*f1*f0)
    literal_if=np.array([1,0,0,0])-1+f1-c*(np.array([0,1,0,0])-1+f0)
    assert abs(true_var-(p1*(1-p1)+c*c*p0*(1-p0)+2*c*p1*p0))<1e-14
    assert abs(np.dot(probs,literal_if)) > .1
    # Appendix (3) versus (4): asymptotic variance of sqrt(N) times estimate.
    mu=.2; lam=.75; G=.25; K=.7
    v_case=mu**2*G*(1-G); v_control=c*c*(1-mu)**2*K*(1-K)
    v3=v_case/(1-lam)+v_control/lam
    v4=v_case/(1-lam)**2+v_control/lam**2
    # N=400, m=100 cases, n=300 controls: Table2code's direct variance.
    assert abs(400*(v_case/100+v_control/300)-v3)<1e-14
    out=dict(status='PASS',scope='Deterministic Python translation and independent algebra; no native MATLAB execution',
             checks=rows,appendix_section3=dict(correct_contribution_variance=true_var,
             literal_printed_variance=literal_var,literal_influence_mean=float(np.dot(probs,literal_if))),
             appendix_case_control=dict(equation3=float(v3),literal_equation4=float(v4)),
             interpretation='Printed-expression discrepancies do not invalidate the centered-contribution MATLAB routines or establish a new ranking result.')
    print(json.dumps(out,indent=2))
    return out

if __name__=='__main__':
    Path(__file__).with_name('source_formula_checks.json').write_text(json.dumps(main(),indent=2)+'\n')
