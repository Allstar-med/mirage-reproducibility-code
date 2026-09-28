from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

P=Path('/workspace/scratch/3ad0c523949b')
navy='#1c344e';teal='#267f83';orange='#c38343';muted='#30465b';grid='#c7d5da';pale='#eef4f4'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(10.8,3.65),gridspec_kw={'width_ratios':[1,1]},sharey=False)
fig.subplots_adjust(left=.14,right=.97,bottom=.26,top=.85,wspace=.46)

def panel(ax,title,labels,counts,colors,limit,tickmax):
 ax.set_title(title,loc='left',fontsize=11,fontweight='bold',color=navy,pad=15)
 ax.barh(range(len(counts)),counts,color=colors,height=.47)
 ax.set_yticks(range(len(labels)),labels,fontsize=8.5,color=navy)
 ax.invert_yaxis();ax.set_xlim(0,limit);ax.set_xticks(tickmax)
 ax.tick_params(axis='x',labelsize=8,colors=muted,length=0)
 ax.tick_params(axis='y',length=0)
 ax.grid(axis='x',color=grid,lw=.8);ax.set_axisbelow(True)
 for spine in ax.spines.values():spine.set_visible(False)
 for y,val in enumerate(counts):ax.text(val+limit*.02,y,f'{val:,}',va='center',fontsize=8.7,color=navy,fontweight='bold')
panel(axes[0],'a  CVAT export',['All exported shapes','Interpolated shapes'],[36041,32922],[navy,orange],43500,[0,10000,20000,30000,40000])
panel(axes[1],'b  Clinician PNG packages',['PNG records','Unique image hashes','Repeated records'],[781,476,305],[navy,teal,orange],950,[0,200,400,600,800])
fig.text(.14,.07,'35 manual-keyframe frames; 91.3% of exported shapes interpolated.',fontsize=9,color=muted)
fig.text(.60,.07,'118 duplicate hash groups crossed v01/v02 prefixes.',fontsize=9,color=muted)
for ext in ['png','pdf']:
 fig.savefig(P/f'Supplementary_Figure_1_High_Contrast_20260928.{ext}',dpi=320,bbox_inches='tight',pad_inches=.10)
print('supp figure done')
