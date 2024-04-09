import pandas as pd
import matplotlib.pyplot as plt
import collections
import sys
title = sys.argv[1]
print(title)
df=pd.read_csv("/home/jasonkjh/works/data/"+title+"/"+title+"_inf1000_plip_analysis_.csv")

Res_Hbond=df["Res_HBonds"]
Res_Pication=df["Res_Pication"]
Res_Pistack=df["Res_Pistack"]
Res_Hydrophobic=df["Res_Hydrophobic"]
Res_Halogen=df["Res_Halogen"]

Hbond=[]
Pication=[]
Pistack=[]
Hphobic=[]
Halogen=[]

for Hbond_ in Res_Hbond:
	if Hbond_ != Hbond_:
		continue
	res=str(Hbond_).split(":")
	for num in res:
		Hbond.append("Hbond:"+num)
for Pcation_ in Res_Pication:
	if Pcation_ != Pcation_:
		continue
	res=str(Pcation_).split(":")
	for num in res:
		Pication.append("Pication:"+num)
for Pstack_ in Res_Pistack:
	if Pstack_ != Pstack_:
		continue
	res=str(Pstack_).split(":")
	for num in res:
		Pistack.append("Pistack:"+num)
for Hphobic_ in Res_Hydrophobic:
	if Hphobic_ != Hphobic_:
		continue
	res=str(Hphobic_).split(":")
	for num in res:
		Hphobic.append("Hphobic:"+num)
for Halogen_ in Res_Halogen:
	if Halogen_ != Halogen_:
		continue
	res=str(Halogen_).split(":")
	for num in res:
		Halogen.append("Halogen:"+num)

dict_Hbond={}
dict_Pication={}
dict_Pistack={}
dict_Hphobic={}
dict_Halogen={}

dict_Hbond=collections.Counter(Hbond)
dict_Pication=collections.Counter(Pication)
dict_Pistack=collections.Counter(Pistack)
dict_Hphobic=collections.Counter(Hphobic)
dict_Halogen=collections.Counter(Halogen)
'''
print(dict_Hbond)
print(dict_Pication)
print(dict_Pistack)
print(dict_Hphobic)
print(dict_Halogen)
'''
dict_=dict_Hbond+dict_Pication+dict_Pistack+dict_Halogen
print(dict_)
dict_=sorted(dict_.items(),key=lambda item:item[1],reverse=True)
key=[]
value=[]
for k,v in dict_:
	key.append(k)
	value.append(v)

idx=0
for v in value:
	if v<50:
		idx=value.index(v)
		break
value=value[:10]
key=key[:10]		
print(value)
fig=plt.figure(figsize=(10,10))
#barlist = plt.bar(key,[1,1,1,1,1,1,1,1,1,1])
barlist=plt.bar(key,value)
#plt.xlabel('Residue',fontsize=15)
def addlabels(x,y):
	for i in range(len(x)):
		plt.text(i,y[i],y[i],ha='center')
for i in range(10):
	if key[i].split(":")[0]=="Hbond":
		barlist[i].set_color('cornflowerblue')
	if key[i].split(":")[0]=="Pistack":
		barlist[i].set_color('yellowgreen')
	if key[i].split(":")[0]=="Halogen":
		barlist[i].set_color('yellow')
	if key[i].split(":")[0]=="Pication":
		barlist[i].set_color('salmon')
#addlabels(value,key)
#for i in range(10):
#	plt.annotate(key[i],xy=(key[i],value[i]),ha='center',va='bottom')
#plt.legend([barlist[0],barlist[3],barlist[5],barlist[8]],['Hydrogen Bond','Pi-Cation Interaction','Pi-Pi stacking','Halogen Interaction'],fontsize=40)
plt.ylabel('Number of ligands',fontsize=25)
plt.xlabel('Residue Number',fontsize=25)
plt.xticks(ticks=key,labels=[x.split(":")[1][0:3] for x in key],fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.savefig("../figures/"+title+"_inf_interaction.png",dpi=600)
