from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt

#from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Lipinski import RotatableBondSmarts
from rdkit.Chem import Fragments
from rdkit.Chem.Descriptors import ExactMolWt
import gc
from gc import collect

title = "PGR"

df=pd.read_csv("/home/jasonkjh/works/data/"+title+"/"+title+"_inf_ucb_substruct.csv")

smi_list=list(df["SMILES"])
#dock_list=list(df["Dock"])
#pred_list=list(df["Pred"])

mol_list=[Chem.MolFromSmiles(smi) for smi in smi_list]

aroma_list=[Chem.rdMolDescriptors.CalcNumAromaticRings(mol) for mol in mol_list]
df["Aroma"]=aroma_list
Hba_list=[Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol) for mol in mol_list]
df["HBA"]=Hba_list
Hbd_list=[Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol) for mol in mol_list]
df["HBD"]=Hbd_list
rot_list=[Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) for mol in mol_list]
df["Rotatable_bond"]=rot_list
NH0_list=[Fragments.fr_NH0(mol) for mol in mol_list]
df["Ter_Amine"]=NH0_list
NH1_list=[Fragments.fr_NH1(mol) for mol in mol_list]
df["Sec_Amine"]=NH1_list
NH2_list=[Fragments.fr_NH2(mol) for mol in mol_list]
df["Pri_Amine"]=NH2_list
Ketone_list=[Fragments.fr_ketone(mol) for mol in mol_list]
df["Ketone"]=Ketone_list
Ester_list= [Fragments.fr_ester(mol) for mol in mol_list]
df["Ester"]=Ester_list
Amide_list=[Fragments.fr_amide(mol) for mol in mol_list]
df["Amide"]=Amide_list
Urea_list=[Fragments.fr_urea(mol) for mol in mol_list]
df["Urea"]=Urea_list
MW_list=[ExactMolWt(mol) for mol in mol_list]
ring_list=[mol.GetRingInfo().NumRings() for mol in mol_list]
df["Molecular Weight"]=MW_list
df["Ring"]=ring_list
df.to_csv("/home/jasonkjh/works/data/"+title+"/"+title+"_inf_ucb_substruct.csv",index=False)

'''
df_0=df[(df["Aroma"]==0)&(df["Dock"]!=0.0)]
pred_0=list(df_0["Dock"])
df_1=df[(df["Aroma"]==1)&(df["Dock"]!=0.0)]
pred_1=list(df_1["Dock"])
df_2=df[(df["Aroma"]==2)&(df["Dock"]!=0.0)]
pred_2=list(df_2["Dock"])
df_3=df[(df["Aroma"]==3)&(df["Dock"]!=0.0)]
pred_3=list(df_3["Dock"])
df_4=df[(df["Aroma"]==4)&(df["Dock"]!=0.0)]
pred_4=list(df_4["Dock"])
df_5=df[(df["Aroma"]==5)&(df["Dock"]!=0.0)]
pred_5=list(df_5["Dock"])
df_6=df[(df["Aroma"]==6)&(df["Dock"]!=0.0)]
pred_6=list(df_6["Dock"])
df_7=df[(df["Aroma"]==7)&(df["Dock"]!=0.0)]
pred_7=list(df_7["Dock"])

df_8=df[(df["Aroma"]==8)&(df["Dock"]!=0.0)]
pred_8=list(df_8["Dock"])


fig=plt.figure(figsize=(15,6))

ax1=plt.subplot(1,2,1)
ax2=plt.subplot(1,2,2)
violin=ax1.violinplot([pred_0,pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7],positions=range(0,8),quantiles=[[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.1,0.9]],showmedians=True)
for vp in violin['bodies']:
	vp.set_facecolor('b')
	vp.set_edgecolor('b')
ax1.set_xlabel('Number of Aromatic ring',fontsize=15)
ax1.set_ylabel('Docking score',fontsize=15)
ax1.set_xticks(range(0,8))
ax1.set_yticks([-2,-3,-4,-5,-6,-7,-8,-9,-10,-11])
ax1.tick_params(axis='both',labelsize=15)

df_0=df[(df["Aroma"]==0)&(df["Dock"]!=0.0)]
pred_0=list(df_0["Pred"])
df_1=df[(df["Aroma"]==1)&(df["Dock"]!=0.0)]
pred_1=list(df_1["Pred"])
df_2=df[(df["Aroma"]==2)&(df["Dock"]!=0.0)]
pred_2=list(df_2["Pred"])
df_3=df[(df["Aroma"]==3)&(df["Dock"]!=0.0)]
pred_3=list(df_3["Pred"])
df_4=df[(df["Aroma"]==4)&(df["Dock"]!=0.0)]
pred_4=list(df_4["Pred"])
df_5=df[(df["Aroma"]==5)&(df["Dock"]!=0.0)]
pred_5=list(df_5["Pred"])
df_6=df[(df["Aroma"]==6)&(df["Dock"]!=0.0)]
pred_6=list(df_6["Pred"])
df_7=df[(df["Aroma"]==7)&(df["Dock"]!=0.0)]
pred_7=list(df_7["Pred"])
df_8=df[(df["Aroma"]==8)&(df["Dock"]!=0.0)]
pred_8=list(df_8["Pred"])
violin=ax2.violinplot([pred_0,pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7],positions=range(0,8),quantiles=[[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.1,0.9]],showmedians=True)
for vp in violin['bodies']:
	vp.set_facecolor('g')
	vp.set_edgecolor('g')

ax2.set_xlabel('Number of Aromatic ring',fontsize=15)
ax2.set_ylabel('Prediction score',fontsize=15)
ax2.set_xticks(range(0,8))
ax2.set_yticks([-2,-3,-4,-5,-6,-7,-8,-9,-10,-11])
ax2.tick_params(axis='both',labelsize=15)

plt.tight_layout(pad=0.9)
plt.savefig("../figures/"+title+"_Aromavsscore.png",dpi=300)
'''
