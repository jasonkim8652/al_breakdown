from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdShapeHelpers
from rdkit.Chem.rdmolfiles import MolFromMol2File
import pandas as pd
import numpy as np
import glob

title = "SOS1"

best_id = sorted(glob.glob("/home/jasonkjh/works/data/"+title+"/"+title+"_dock1000/"+"*.mol2"))
rnd_id = sorted(glob.glob("/home/jasonkjh/works/data/"+title+"/"+title+"_rnd1000/"+"*.mol2"))
worst_id = sorted(glob.glob("/home/jasonkjh/works/data/"+title+"/"+title+"_worst1000/"+"*.mol2"))

best_id = [i.split("/")[-1].split(".")[0].split("_")[0]+ "_"+i.split("/")[-1].split(".")[0].split("_")[1]for i in best_id]
rnd_id = [i.split("/")[-1].split(".")[0].split("_")[0]+ "_"+i.split("/")[-1].split(".")[0].split("_")[1] for i in rnd_id]
worst_id = [i.split("/")[-1].split(".")[0].split("_")[0]+ "_"+i.split("/")[-1].split(".")[0].split("_")[1] for i in worst_id]

'''
id_list = []
df= pd.read_csv("/home/jasonkjh/works/active_learning/EGFR_best.csv")

id_list.extend(list(df["ID"]))
df=pd.read_csv("/home/jasonkjh/works/active_learning/EGFR_rnd.csv")

id_list.extend(list(df["ID"]))
df=pd.read_csv("/home/jasonkjh/works/active_learning/EGFR_worst.csv")
id_list.extend(list(df["ID"]))
'''
#id_list=list(set(id_list))
#id_list=sorted(id_list)

print("Best")
print(len(best_id))
print(best_id[0])
print("Rnd")
print(len(rnd_id))
print(rnd_id[0])
print("Worst")
print(len(worst_id))
print(worst_id[0])

dist_matrix=[]

id_list = best_id + rnd_id + worst_id

for i in id_list:
    dist_row = []
    for j in id_list:
        try:
            if i in best_id:
                mol1 = MolFromMol2File("/home/jasonkjh/works/data/"+title+"/"+title+"_dock1000/"+i+"_docked.mol2")
            elif i in rnd_id:
                mol1 = MolFromMol2File("/home/jasonkjh/works/data/"+title+"/"+title+"_rnd1000/"+i+"_docked.mol2")
            elif i in worst_id:
                mol1 = MolFromMol2File("/home/jasonkjh/works/data/"+title+"/"+title+"_worst1000/"+i+"_docked.mol2")
            if j in best_id:
                mol2 = MolFromMol2File("/home/jasonkjh/works/data/"+title+"/"+title+"_dock1000/"+j+"_docked.mol2")
            elif j in rnd_id:
                mol2 = MolFromMol2File("/home/jasonkjh/works/data/"+title+"/"+title+"_rnd1000/"+j+"_docked.mol2")
            elif j in worst_id:
                mol2 = MolFromMol2File("/home/jasonkjh/works/data/"+title+"/"+title+"_worst1000/"+j+"_docked.mol2")
                
			#o3a = rdMolAlign.GetO3A(mol1, mol2)
            dist =  rdShapeHelpers.ShapeTanimotoDist(mol1, mol2)
            print(str(i)+","+str(j))
            print(dist)
            dist_row.append(dist)
        except:
            print("error_handling")
            print(str(i) +","+str(j))
            dist_row.append(2.0)
    dist_matrix.append(dist_row)
dist_matrix=np.array(dist_matrix)
np.save("/home/jasonkjh/works/data/"+title+"/"+title+"_dist_best_rnd_worst.npy",dist_matrix)
