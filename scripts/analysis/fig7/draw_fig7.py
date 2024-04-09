import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
import pandas as pd
#from umap.umap_ import UMAP
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
import glob
import sys

title = sys.argv[1]
best_num = 1000
dist_matrix = np.load("/home/jasonkjh/works/data/"+title+"/"+title+"_dist_best_rnd_worst.npy")
id_list = []
best_id= sorted(glob.glob("/home/jasonkjh/works/data/"+title+"/"+title+"_dock1000/*.mol2"))
rnd_id=sorted(glob.glob("/home/jasonkjh/works/data/"+title+"/"+title+"_rnd1000/*.mol2"))
worst_id=sorted(glob.glob("/home/jasonkjh/works/data/"+title+"/"+title+"_worst1000/*.mol2"))

#using the rnd_id, extract rnd_id from /home/jasonkjh/works/data/Enamine_HTS/Enamine_hts_collection.csv and save it as title+"_rnd1000.csv", change the SMILES_canonical column name to SMILES


best_id = [i.split("/")[-1].split(".")[0].split("_")[0]+ "_"+i.split("/")[-1].split(".")[0].split("_")[1]for i in best_id]
rnd_id = [i.split("/")[-1].split(".")[0].split("_")[0]+ "_"+i.split("/")[-1].split(".")[0].split("_")[1] for i in rnd_id]
worst_id = [i.split("/")[-1].split(".")[0].split("_")[0]+ "_"+i.split("/")[-1].split(".")[0].split("_")[1] for i in worst_id]

print(len(best_id))
print(len(rnd_id))
print(len(worst_id))


# Step 1: Read the file
df = pd.read_csv('/home/jasonkjh/works/data/Enamine_HTS/Enamine_hts_collection.csv')

# Assuming rnd_id is a list of IDs you want to extract
# Step 2: Extract rows based on rnd_id
extracted_df = df[df['ID'].isin(rnd_id)]

# Step 3: Rename the column
extracted_df = extracted_df.rename(columns={"SMILES_canonical": "SMILES","SMILES":"SMILES_old"})

# Step 4: Save the extracted data
filename = title + "_rnd1000.csv"
extracted_df.to_csv('/home/jasonkjh/works/data/'+title+'/'+filename, index=False)

id_list = best_id+rnd_id+worst_id

#id_list = list(set(id_list))
#id_list = sorted(id_list)

best_idx= []
rnd_idx= []
worst_idx = []

'''
for id_ in list(df_best["ID"])[:50]:
	best_idx.append(id_list.index(id_))

for id_ in list(df_best["ID"])[50:]:
        rnd_idx.append(id_list.index(id_))

for id_ in list(df_rnd["ID"]):
	rnd_idx.append(id_list.index(id_))

for id_ in list(df_worst["ID"]):
        rnd_idx.append(id_list.index(id_))
'''
#print(best_idx)
#print(rnd_idx)

print(dist_matrix.shape)
#print(dist_matrix)

dist_matrix = (dist_matrix+dist_matrix.T)/2

'''
int_idx = []
int_idx.extend(best_idx)
int_idx.extend(rnd_idx)
'''
df_best = pd.read_csv("/home/jasonkjh/works/data/"+title+"/"+title+"_dock1000.csv")[:best_num]
df_rnd = pd.read_csv("/home/jasonkjh/works/data/"+title+"/"+title+"_rnd1000.csv")
df_worst = pd.read_csv("/home/jasonkjh/works/data/"+title+"/"+title+"_worst1000.csv")

smi_list = []
best_idx = []
rnd_idx =[]
worst_idx = []
for i,id_ in enumerate(id_list):
	if id_ in df_best["ID"].values:
		smi_list.append(list(df_best[df_best["ID"]==id_]["SMILES"])[0])
		best_idx.append(i)
	elif id_ in df_rnd["ID"].values:
		smi_list.append(list(df_rnd[df_rnd["ID"]==id_]["SMILES"])[0])
		rnd_idx.append(i)
	elif id_ in df_worst["ID"].values:
		smi_list.append(list(df_worst[df_worst["ID"]==id_]["SMILES"])[0])
		worst_idx.append(i)
	else:
		continue
#df_all = pd.concat([df_best, df_rnd, df_worst])
#df_all = df_all.reset_index(drop=True)
print(best_idx)
print(rnd_idx)
print(worst_idx)
idx_ = best_idx+rnd_idx+worst_idx
dist_matrix = np.take(dist_matrix,idx_,axis = 0)
dist_matrix = np.take(dist_matrix,idx_,axis = 1)
print(dist_matrix.shape)

best_idx_in_mat = range(len(best_idx))
rnd_idx_in_mat = range(len(best_idx),len(best_idx)+len(rnd_idx))
worst_idx_in_mat = range(len(best_idx)+len(rnd_idx),len(best_idx)+len(rnd_idx)+len(worst_idx))

mw_list = np.array([MolWt(Chem.MolFromSmiles(smi)) for smi in smi_list])
#mw = df_all["SMILES"].apply(lambda smi: MolWt(Chem.MolFromSmiles(smi)))
a_ = 1.1

X = 1.0 / (a_ - dist_matrix) / np.sqrt(np.outer(mw_list, mw_list))
MDS_model = manifold.MDS(n_components = 2, n_jobs = 48, dissimilarity = "precomputed")
print("fitting start!")
MDS_fit = MDS_model.fit(X)
print("fitting finished!")
MDS_coords = MDS_model.fit_transform(X)
#best_coords1 = MDS_coords[best_idx,0]
#best_coords2 = MDS_coords[best_idx,1]
#rnd_coords1 = MDS_coords[rnd_idx,0]
#rnd_coords2 = MDS_coords[rnd_idx,1]
# Calculate the center of the 'Best' group
best_center_x = np.mean(MDS_coords[best_idx_in_mat, 0])
best_center_y = np.mean(MDS_coords[best_idx_in_mat, 1])

# Determine the spread from the center to the farthest point in the 'Best' group
max_spread_x = np.max(np.abs(MDS_coords[best_idx_in_mat, 0] - best_center_x))
max_spread_y = np.max(np.abs(MDS_coords[best_idx_in_mat, 1] - best_center_y))
max_spread = max(max_spread_x, max_spread_y)

# Ensure the plot includes a margin around the 'Best' group
margin = 1.1 * max_spread

# Set the axis limits to center the 'Best' group
plt.axis([best_center_x - margin, best_center_x + margin, best_center_y - margin, best_center_y + margin])

# Continue with the plotting as before
plt.figure()
plt.scatter(MDS_coords[worst_idx_in_mat,0], MDS_coords[worst_idx_in_mat,1], alpha=0.1, c='yellowgreen', label='Worst')
plt.scatter(MDS_coords[rnd_idx_in_mat,0], MDS_coords[rnd_idx_in_mat,1], alpha=0.1, c='darkblue', label='Random')
plt.scatter(MDS_coords[best_idx_in_mat,0], MDS_coords[best_idx_in_mat,1], alpha=0.1, c='orangered', label='Best')

leg = plt.legend(fontsize=25)
for lh in leg.legendHandles: 
    lh.set_alpha(1)

# It might not be necessary to adjust the axes based on min and max of the coordinates anymore
# since we're focusing on centering the 'Best' group. However, you can adjust if needed.

plt.xticks(color='w')
plt.yticks(color='w')
plt.savefig(title+"_MDS/LAST.png", dpi=600)
    #dist_matrix = dist_matrix+np.identity(dist_matrix.shape[0])
	#X = np.take(X,int_idx,axis = 0)
	#X = np.take(X,int_idx,axis = 1)
#print(np.sort(np.ravel(X))[:15])
#print(np.max(X))
#print(np.min(X))



'''
	id_list = []
	df_best= pd.read_csv("/home/jasonkjh/works/active_learning/6SCM_best.csv")[:50]

	id_list.extend(list(df_best["ID"]))
	df_rnd=pd.read_csv("/home/jasonkjh/works/active_learning/6SCM_rnd.csv")
	id_list.extend(list(df_rnd["ID"]))

	best_idx= []
	rnd_idx= []
	worst_idx = []
	id_list= id_list
	for id_ in list(df_best["ID"])[:50]:
		best_idx.append(id_list.index(id_))
	for id_ in list(df_rnd["ID"]):
		rnd_idx.append(id_list.index(id_))
	rnd_idx = list(set(rnd_idx)-set(best_idx))
	print(rnd_idx)
	print(len(best_idx))
	print(len(rnd_idx))
	#umap_model = UMAP(n_components =2, n_neighbors = neighbor, metric = "precomputed", min_dist = dist )
'''
   
#umap_coords = umap_model.fit_transform(X)

	
#print(str(neighbor)+","+str(dist))
#print(np.std(best_coords1)/np.std(rnd_coords1))
#print(np.std(best_coords2)/np.std(rnd_coords2))
	
#plt.scatter(umap_coords[worst_idx,0],umap_coords[worst_idx,1],alpha=0.5,c='orangered',label='Worst')
	

