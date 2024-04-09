from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
title = "PGR"
# Sample DataFrames: Replace with your actual DataFrames
df1_active = pd.read_csv('/home/jasonkjh/works/data/'+title+'/'+title+'_active_dock.csv')
df1_decoy = pd.read_csv('/home/jasonkjh/works/data/'+title+'/'+title+'_decoy_dock.csv')
df1_active["active"] = True
df1_decoy["active"] = False
df1 = pd.concat([df1_active, df1_decoy])
df2_active = pd.read_csv('/home/jasonkjh/works/data/'+title+'/'+title+'_active_inf_greedy.csv')
df2_decoy = pd.read_csv('/home/jasonkjh/works/data/'+title+'/'+title+'_decoy_inf_greedy.csv')
df2_active["active"] = True
df2_decoy["active"] = False
df2 = pd.concat([df2_active, df2_decoy])
df3_active = pd.read_csv('/home/jasonkjh/works/data/'+title+'/'+title+'_active_inf_ucb.csv')
df3_decoy = pd.read_csv('/home/jasonkjh/works/data/'+title+'/'+title+'_decoy_inf_ucb.csv')
df3_active["active"] = True
df3_decoy["active"] = False
df3 = pd.concat([df3_active, df3_decoy])
df4_active = pd.read_csv('/home/jasonkjh/works/data/'+title+'/'+title+'_active_inf_unc.csv')
df4_decoy = pd.read_csv('/home/jasonkjh/works/data/'+title+'/'+title+'_decoy_inf_unc.csv')
df4_active["active"] = True
df4_decoy["active"] = False
df4 = pd.concat([df4_active, df4_decoy])

# Initialize the plot
plt.figure()

fpr, tpr, _ = roc_curve(df1['active'], -df1['Dock'])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2)

fpr, tpr, _ = roc_curve(df2['active'], -df2['Pred'])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2)

fpr, tpr, _ = roc_curve(df3['active'], -df3['Pred'])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2)

fpr, tpr, _ = roc_curve(df4['active'], -df4['Pred'])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2) 



# Configure plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=20)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.yticks([0.2,0.4,0.6,0.8,1.0],fontsize=20)
#plt.title('Receiver Operating Characteristic')
#plt.legend(loc='lower right')
plt.tight_layout()
# Display plot
plt.savefig("../figures/"+title+"_roc.png",dpi=600)
