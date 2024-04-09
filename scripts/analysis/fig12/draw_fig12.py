import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.ML.Cluster import ClusterVis
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.manifold import TSNE
from scipy.stats import ttest_ind
import sys 

title = sys.argv[1]

df=pd.read_csv("/home/jasonkjh/works/data/"+title+"/"+title+"_real_inf_dock.csv")

inf=list(df[df['Dock']<0]['Dock'])

df2=pd.read_csv("/home/jasonkjh/works/data/"+title+"/"+title+"_real_fp_dock.csv")

fp=list(df2[df2['Dock']<0]['Dock'])

df3=pd.read_csv("/home/jasonkjh/works/data/"+title+"/"+title+"_real_rand_dock.csv")

rand = list(df3[df3['Dock']<0]['Dock'])

stat , p_value_fp = ttest_ind(inf, fp)

print(f'P-value between INF and FP: {p_value_fp}')

stat, p_value_rand = ttest_ind(inf,rand)

print(f'P-value between INF and RAND: {p_value_rand}')

plt.figure(figsize=(10, 6))

colors = ['darkblue', 'orangered', 'yellowgreen']  # Colors for each group
data_groups = [fp, inf, rand]
labels = ['FP Screening', 'Inference', 'Random']  # Labels for x-ticks

max_value_fp = max(max(fp), max(inf))  # Calculate max value for positioning
max_value_rand = max(max(rand), max(inf))

for i, data in enumerate(data_groups):
    vp = plt.violinplot(data, positions=[i], showmedians=True, widths=0.9)
    for pc in vp['bodies']:
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
    # No need to iterate over LineCollection objects directly

# Add custom x-tick labels
plt.xticks(ticks=[0, 1, 2], labels=labels, fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("Docking Score",fontsize=25)
# Drawing a line and positioning p-value text
line_y_fp = max_value_fp * 0.8  # Position for the line and text
line_y_rand = max_value_rand * 0.8
plt.plot([1, 2], [line_y_rand, line_y_rand], color='black', lw=1)  # Draw connecting line
plt.plot([0, 1], [line_y_fp, line_y_fp], color='black',lw=1)
plt.vlines(x = 0, ymin = max(fp), ymax = line_y_fp, colors ='black', lw = 1)
plt.vlines(x = 1, ymin = max(inf), ymax = line_y_rand, colors ='black', lw = 1)
plt.vlines(x = 2, ymin = max(rand), ymax = line_y_rand, colors = 'black', lw = 1)
plt.text(1.5, line_y_rand*1.01, '***', ha='center', va='bottom', fontsize=25)  # Position text above line
if p_value_fp>0.05:
    plt.text(0.5, line_y_fp*0.99, 'ns',ha='center', va= 'bottom', fontsize=25)
elif (p_value_fp<=0.05)&(p_value_fp>0.01):
    plt.text(0.5, line_y_fp*1.05, '*',ha='center', va= 'bottom', fontsize=25)
elif (p_value_fp<=0.01)&(p_value_fp>0.001):
    plt.text(0.5, line_y_fp*1.05, '**',ha='center', va= 'bottom', fontsize=25)
elif (p_value_fp<0.001):
    plt.text(0.5, line_y_fp*1.01, '***',ha='center', va= 'bottom', fontsize=25)
plt.tight_layout()
plt.savefig('../figures/'+title+'_matchinginf.png', dpi=600)
plt.close()

def embedding(smiles):
    materials_data=[Chem.MolFromSmiles(x) for x in smiles]
    fingerprints=[Chem.RDKFingerprint(x) for x in materials_data]
    return fingerprints

smi_inf = list(df[df['Dock']<0]['SMILES'])
smi_fp = list(df2[df2['Dock']<0]['SMILES'])
smi_rand = list(df3[df3['Dock']<0]['SMILES'])

smiles_data = smi_inf + smi_fp + smi_rand

fingerprints=embedding(smiles_data)
fingerprints = np.array(fingerprints)

model=TSNE(n_components=2,n_jobs=48)
Xt=model.fit_transform(fingerprints)
plt.scatter(Xt[len(df):len(df2)+len(df),0],Xt[len(df):len(df2)+len(df),1],alpha = 0.2,color='darkblue')
plt.scatter(Xt[:len(df),0],Xt[:len(df),1],alpha = 0.2, color = 'orangered')
plt.scatter(Xt[len(df)+len(df2):len(df)+len(df2)+len(df3),0],Xt[len(df)+len(df2):len(df)+len(df2)+len(df3),1],alpha = 0.2,color='yellowgreen')
#leg = plt.legend(fontsize = 15)
#for lh in leg.legendHandles:
#    lh.set_alpha(1)
plt.xticks(color='w')
plt.yticks(color='w')
plt.tight_layout()
plt.savefig("../figures/"+title+"_tsne.png",dpi =600)

