import io, zipfile, os, subprocess
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from docx import Document

O=Path(__file__).resolve().parent
navy='#20344d'; blue='#277b86'; gold='#b87924'; muted='#30465b'; light='#cbd7dc'; faint='#f4f6f7'; green='#489d89'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
def save(fig,n):
 fig.savefig(O/(n+'.pdf'),bbox_inches='tight',pad_inches=.12)
 subprocess.run(['pdftoppm','-f','1','-singlefile','-png','-r','300',str(O/(n+'.pdf')),str(O/n)],check=True)
 plt.close(fig)

# Fig 2: matched test robust effect and sensitivity controls.
fig=plt.figure(figsize=(11.8,5.55));gs=fig.add_gridspec(2,2,left=.10,right=.97,bottom=.13,top=.92,wspace=.38,hspace=.48,width_ratios=[1.05,1.55])
a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[:,1]);c=fig.add_subplot(gs[1,0])
a.set_title('a  Paired models on the same 598 frames',loc='left',fontweight='bold',color=navy,fontsize=11)
for x,y in zip([.674,.698,.670],[.573,.595,.542]):a.plot([0,1],[x,y],color='#586d79',lw=1.8,zorder=1)
a.scatter([0]*3,[.674,.698,.670],c=blue,s=46,zorder=2);a.scatter([1]*3,[.573,.595,.542],c=gold,s=46,zorder=2)
a.set_xticks([0,1],['Earlier target\noperation','Different\noperation']);a.set_xlim(-.22,1.22);a.set_ylim(.47,.77);a.set_ylabel('Macro AUROC',color=navy)
a.grid(axis='y',color=light,lw=.7);a.set_axisbelow(True)
b.set_title('b  Direction retained across prespecified and sensitivity analyses',loc='left',fontweight='bold',color=navy,fontsize=11)
vals=[(.110717,.017986,.186086,'Frozen primary','3 epochs'),(.125952,.068942,.175141,'Longer training','12 epochs'),(.103307,.032489,.159653,'Measured-source adjustment','12 epochs')]
for j,(x,lo,hi,lab,detail) in enumerate(vals):
 y=2-j;c0=navy if j==0 else blue
 b.plot([lo,hi],[y,y],lw=2.4,color=c0);b.plot([lo,lo],[y-.06,y+.06],color=c0,lw=1.5);b.plot([hi,hi],[y-.06,y+.06],color=c0,lw=1.5);b.scatter([x],[y],s=76,c=c0,zorder=3)
 b.text(-.012,y+.18,lab,color=navy,fontsize=10,fontweight='bold',va='center');b.text(-.012,y-.2,detail,color=muted,fontsize=9.1,va='center')
 b.text(.193,y,f'{x:.3f}  [{lo:.3f}, {hi:.3f}]',color=navy,fontsize=9,va='center')
b.axvline(0,color='#78909b',lw=1);b.set_xlim(-.02,.35);b.set_ylim(-.56,2.54);b.set_yticks([]);b.set_xlabel('Paired macro AUROC difference  (same minus different)',color=navy);b.grid(axis='x',color=light,lw=.7);b.set_axisbelow(True)
c.set_title('c  No single operation determines the contrast',loc='left',fontweight='bold',color=navy,fontsize=11)
s=Document(O/'Supplementary_Information_Final_20260928.docx');t=next(t for t in s.tables if t.cell(0,0).text=='Omitted operation')
v=np.array([float(row.cells[1].text) for row in t.rows[1:]])
c.scatter(np.arange(1,21),np.sort(v),c=blue,s=25);c.axhline(.110717,color=gold,lw=1.4,ls='--')
c.set_xlim(.5,20.5);c.set_ylim(.06,.14);c.set_ylabel('Leave-one-operation-out Δ',color=navy);c.set_xlabel('20 operations, ranked by Δ');c.grid(axis='y',color=light,lw=.7);c.set_axisbelow(True)
fig.text(.10,.025,'95% intervals resample complete operations and condition on the trained models; source adjustment does not control clinical phase.',fontsize=9.4,color=muted)
save(fig,'Figure_2_High_Contrast_20260928')

# Fig 3: 40 actual operation identities as the central visual unit.
with zipfile.ZipFile(O/'Mirage_reproducibility_code_and_derived_results_20260928.zip') as z:
 d=pd.read_csv(io.BytesIO(z.read('Evidence/Locked_Test/seed17/locked_test_predictions.csv')))
g=d.groupby('video',sort=True);ids=list(g.groups);counts=g.size().to_numpy();states=np.zeros((3,len(ids)),int)
for j,col in enumerate(['y_c1','y_c2','y_c3']):
 for i,(_,sub) in enumerate(g):
  pos=sub[col].sum();states[j,i]=2 if 0<pos<len(sub) else (1 if pos else 0)
assert len(ids)==40 and [sum(row==2) for row in states]==[24,19,22]
fig=plt.figure(figsize=(12.2,5.6));gs=fig.add_gridspec(3,1,left=.08,right=.98,bottom=.15,top=.92,hspace=.48,height_ratios=[.55,1.65,1.25])
a=fig.add_subplot(gs[0]);b=fig.add_subplot(gs[1],sharex=a);c=fig.add_subplot(gs[2],sharex=a)
a.set_title('a  1,663 labelled frames are nested in 40 unseen operations',loc='left',color=navy,fontweight='bold',fontsize=12)
a.bar(np.arange(40),counts,width=.82,color='#7d99a4');a.set_ylim(0,max(counts)*1.18);a.set_ylabel('Frames',fontsize=9);a.spines[['bottom','left']].set_visible(False);a.tick_params(left=False,labelleft=False,bottom=False,labelbottom=False)
from matplotlib.colors import ListedColormap
cmap=ListedColormap(['#bacbd1','#d19a4d',blue]);b.imshow(states,aspect='auto',interpolation='none',cmap=cmap,vmin=0,vmax=2,extent=(-.5,39.5,2.5,-.5))
b.set_yticks(range(3),['C1','C2','C3']);b.set_ylabel('CVS criterion',fontsize=9);b.set_xlim(-.5,39.5);b.tick_params(axis='x',bottom=False,labelbottom=False);b.spines['bottom'].set_visible(False)
for x in range(41):b.axvline(x-.5,color='white',lw=.6)
for y in range(4):b.axhline(y-.5,color='white',lw=.6)
# Number of estimable operations is visually the key result.
for j,n in enumerate([24,19,22]):b.text(39.7,j,f'{n}/40 with both states',ha='left',va='center',color=navy,fontsize=9.4,clip_on=False)
for ax in [a,b,c]:ax.spines[['top','right']].set_visible(False)
c.set_title('b  Operation-level estimability differs by criterion',loc='left',color=navy,fontweight='bold',fontsize=11)
for j,(n,p) in enumerate(zip([24,19,22],[.251,.177,.299])):
 yy=2-j;c.barh(yy,n,height=.38,color=blue);c.plot([n,40],[yy,yy],color=light,lw=8,solid_capstyle='butt');c.text(n+.8,yy,f'{n} / 40',va='center',color=navy,fontsize=9)
c.set_xlim(-.5,46);c.set_ylim(-.7,2.8);c.set_yticks([2,1,0],['C1','C2','C3']);c.set_xlabel('Operations allowing within-operation AUROC',fontsize=9);c.set_xticks([0,10,20,30,40]);c.grid(axis='x',color=light,lw=.7);c.set_axisbelow(True)
fig.legend(handles=[Line2D([0],[0],marker='s',linestyle='',color='w',markerfacecolor='#bacbd1',markersize=9,label='Negative only'),Line2D([0],[0],marker='s',linestyle='',color='w',markerfacecolor='#d19a4d',markersize=9,label='Positive only'),Line2D([0],[0],marker='s',linestyle='',color='w',markerfacecolor=blue,markersize=9,label='Both states')],ncol=3,loc='lower left',bbox_to_anchor=(.08,.018),frameon=False,fontsize=9)
save(fig,'Figure_3_High_Contrast_20260928')
print('Done',len(d),len(ids))
