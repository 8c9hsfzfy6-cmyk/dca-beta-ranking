from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]/'dca_round_final';D=pd.read_csv(R/'analysis/verified_results.csv');P=pd.read_csv(R/'results/population/population_grid.csv')
names={'beta_estimated_mean':'Estimated-mean Beta','risk_only':'Risk-only','outcome_based':'Outcome-based','beta_known_mean':'Known-mean Beta'}
colors=['#C35438','#237F7B','#355B9A','#9467A5']
E=D[(D.scenario_id==3)&(abs(D.threshold-1/9)<1e-12)]
fig,axs=plt.subplots(1,2,figsize=(10.5,4.2))
for i,(m,label) in enumerate(names.items()):
 a=E[E.method==m].sort_values('n');x=np.arange(3)+(i-1.5)*.035
 axs[0].errorbar(x,a.strict_wrong_rate*100,yerr=np.array([a.strict_wrong_rate-a.wrong_wilson_low,a.wrong_wilson_high-a.strict_wrong_rate])*100,marker=['o','s','^','D'][i],linestyle=['-','--','-.',':'][i],lw=1.3,capsize=2,color=colors[i],label=label)
 axs[1].errorbar(x,a.mean_regret*1000,yerr=1.96*a.regret_mcse*1000,marker=['o','s','^','D'][i],linestyle=['-','--','-.',':'][i],lw=1.3,capsize=2,color=colors[i])
for ax in axs:
 ax.set_xticks(range(3),['500','3,000','10,000']);ax.set_xlabel('Evaluation sample size');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
axs[0].set_ylabel('Strict wrong ordering (%)');axs[0].set_ylim(-3,105);axs[0].set_title('A  Ordering errors')
axs[1].set_ylabel('Mean net-benefit loss × 1,000');axs[1].set_title('B  Decision loss');axs[1].set_ylim(bottom=0)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=9,frameon=False);fig.tight_layout(rect=(0,.10,1,1));fig.savefig(Path(__file__).resolve().parents[1]/'figures/Figure_2.png',dpi=400);plt.close(fig)
