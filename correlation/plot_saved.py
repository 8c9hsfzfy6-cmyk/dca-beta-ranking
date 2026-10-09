from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
DATA=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results'
summary=pd.read_csv(DATA/'primary_summary.csv')
labels={'beta_estimated_mean':'Beta estimated mean','risk_only':'Risk only',
        'outcome_based':'Outcome based','beta_known_mean':'Beta known mean'}
styles=[('0.05','o','-'),('0.4','s','--'),('0.6','^','-.'),('0.2','D',':')]
names=['beta_control','low_endpoint','high_auc']
titles=['Correct Beta family','Low endpoint mass','High AUC']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axs=plt.subplots(3,2,figsize=(8.8,9.6),layout='constrained')
for i,name in enumerate(names):
    sub=summary[(summary['case']==name)&(summary.n==10000)&summary.rho0.eq(summary.rho1)]
    for method,(color,marker,linestyle) in zip(labels,styles):
        q=sub[sub.method.eq(method)].sort_values('rho0')
        for col,mean,lo,hi,scale in [(0,'strict_wrong_rate','wrong_low','wrong_high',100),
                                     (1,'mean_loss','loss_low','loss_high',1000)]:
            err=np.maximum(0,np.vstack([q[mean]-q[lo],q[hi]-q[mean]]))*scale
            axs[i,col].errorbar(q.rho0,q[mean]*scale,yerr=err,color=color,marker=marker,
                linestyle=linestyle,markersize=4,linewidth=1.1,capsize=2,label=labels[method])
    axs[i,0].set_title(f'{chr(65+2*i)}  {titles[i]}')
    axs[i,1].set_title(f'{chr(66+2*i)}  {titles[i]}')
    axs[i,0].set_ylabel('Strict wrong ordering (%)');axs[i,0].set_ylim(-3,103)
    axs[i,1].set_ylabel('Expected NB loss per 1000')
    for ax in axs[i]:
        ax.set_xticks([-.5,0,.3,.6,.9]);ax.grid(alpha=.2)
        ax.set_xlabel('Conditional latent correlation')
axs[0,0].legend(fontsize=7.5,loc='upper right')
figpath=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT.parent/'figures'/'Figure_S4.png';figpath.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(figpath,dpi=300);plt.close(fig)

