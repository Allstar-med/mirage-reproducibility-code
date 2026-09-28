from pathlib import Path
from io import BytesIO
import fitz
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

P=Path('/workspace/scratch/3ad0c523949b')
navy='#172b40'; teal='#1f7883'; amber='#bd802f'; muted='#30465b'; rule='#b8c8ce'; soft='#eef4f4'; warm='#f9f2e9'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'savefig.facecolor':'white'})
fig,ax=plt.subplots(figsize=(12.4,7.1));fig.subplots_adjust(left=.03,right=.97,bottom=.04,top=.97)
ax.set_xlim(0,12.4);ax.set_ylim(0,7.1);ax.axis('off')
def txt(x,y,s,size=10,color=navy,weight=None,ha='left',va='center'):
 ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va=va)
def line(x0,x1,y,c=rule,w=1):ax.plot([x0,x1],[y,y],color=c,lw=w,solid_capstyle='butt')

# The surgical thumbnails are extracted from the prior vector figure; no new photographic content.
def strip(path):
 doc=fitz.open(path);record=doc.extract_image(doc[0].get_images(full=True)[0][0]);return Image.open(BytesIO(record['image'])).convert('RGB')
same=strip(P/'Figure_1_Evaluation_geometry.pdf');mixed=strip(P/'Figure_1_Claim_split_audit_corrected.pdf')
frames=[same.crop((0,0,125,75)),same.crop((138,0,265,75)),same.crop((276,0,405,75)),mixed.crop((370,0,470,62))]

txt(.24,6.88,'a',13,teal,'bold');txt(.50,6.88,'Frames remain related within an operation',12,navy,'bold')
txt(6.42,6.88,'b',13,teal,'bold');txt(6.68,6.88,'One fixed test, two training histories',12,navy,'bold')
line(.24,5.99,6.57);line(6.42,12.14,6.57)
for i,im in enumerate(frames):
 x=.25+1.17*i if i<3 else 4.49
 ax.imshow(im,extent=(x,x+1.0,5.40,6.05),aspect='auto',interpolation='lanczos',zorder=1)
 ax.add_patch(Rectangle((x,5.40),1,.65,fill=False,edgecolor='white',lw=.8,zorder=2))
ax.plot([.25,3.59],[5.28,5.28],color=teal,lw=2.4)
txt(1.91,5.12,'Operation 170',9.2,teal,'bold',ha='center')
ax.plot([3.91,3.91],[5.23,6.08],color=amber,lw=1.6,ls=(0,(3,3)))
txt(4.99,5.12,'Operation 171',9.2,amber,'bold',ha='center')
txt(.25,4.79,'Frame order alone does not establish elapsed time.',9.0,muted)

# Scores are aligned numerically and the common test is explicit.
for y,bg,label,score,col in [(5.78,soft,'Earlier frames from target operations','0.681',teal),
                              (5.05,warm,'Frames from different operations','0.570',amber)]:
 ax.add_patch(Rectangle((6.46,y-.29),4.40,.54,facecolor=bg,edgecolor='none'))
 ax.add_patch(Rectangle((6.46,y-.29),.065,.54,facecolor=col,edgecolor='none'))
 txt(6.66,y,label,9.2)
 txt(11.11,y,score,13,col,'bold',ha='center')
txt(9.66,4.47,'Same 598 test frames · 20 operations',9.1,navy,'bold',ha='center')
txt(11.63,4.47,'Δ 0.111',11.4,navy,'bold',ha='center')
line(6.47,12.12,4.30)

txt(.24,3.99,'c',13,teal,'bold');txt(.50,3.99,'The split and uncertainty unit determine the claim',12,navy,'bold')
xs=[.34,4.04,7.33,10.03]
heads=['Intended claim','Training–test separation','Inference unit','Evidence here']
for x,h in zip(xs,heads):txt(x,3.54,h,9.7,navy,'bold')
line(.28,12.13,3.30,rule,1.15)
rows=[
 ('Another frame','Frame or clip','Frame cluster','Illustrated only'),
 ('Later frames, same operation','Keyframe-index buffer','Operation','Paired test'),
 ('New operation','Operation identity','Operation','Official test'),
 ('New patient','Linked patient identity','Patient','Not assessed'),
 ('New institution','Institution identity','Site','Not assessed'),
]
for i,row in enumerate(rows):
 y=3.03-i*.53
 if i==2:ax.add_patch(Rectangle((.28,y-.25),11.85,.50,facecolor=soft,edgecolor='none'))
 for j,(x,value) in enumerate(zip(xs,row)):
  col=teal if i==2 else muted if i>2 else navy
  txt(x,y,value,9.55,col,'bold' if i==2 else None)
 line(.28,12.13,y-.27,'#cad6da',.7)
txt(.34,.22,'The paired contrast is observational: training-source content and clinical phase were not fully matched.',9.05,muted)
out=P/'Figure_1_High_Contrast_20260928'
fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.08)
fig.savefig(out.with_suffix('.png'),dpi=300,bbox_inches='tight',pad_inches=.08)
print(out)
