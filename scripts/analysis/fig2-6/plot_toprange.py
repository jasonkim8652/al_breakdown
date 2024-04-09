import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

title = 'PGR'

def main():
	#seed_list = [i for i in range(4)]
	#step_list = [i for i in range(10)]
	method_list = ['greedy', 'ucb', 'unc']
	#method_list = ['greedy','ucb']
	#title = 'trial'
	num_search = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
	#df_all = pd.read_csv('/home/seongok/works/active_learning/results/6SCM_Enamine_final.csv')
	contents=[]
	for method in method_list:
		val_list=[]
		df_all=pd.read_csv('/home/jasonkjh/works/data/'+title+'/'+title+'_inf_'+method+'.csv')
		for topk in num_search:
			df_topk = df_all.sort_values(by=['Dock'], ascending=True)[topk-1000:topk]
			id_topk = list(df_topk['ID'])
			print (len(id_topk))
			df_trial = df_all.sort_values(by=['Pred'], ascending=True)[topk-1000:topk]
			id_trial = list(df_trial['ID'])
			num_found = len(list(set(id_topk) & set(id_trial)))
			success_rate = 100.0 * num_found/topk
			val_list.append(success_rate)
		contents.append(val_list)
		#contents_list.append(contents)

	'''
	contents_list = []
	for method in method_list:
		contents = []
		for seed in seed_list:
			val_list = []
			for step in step_list:
				path = '/home/jasonkjh/works/active_learning/data/'
				path += title 
				path += '_seed' + str(seed)
				path += '_step' + str(step)
				if step != 0:
					path += '_' + method 
				path += '_train.csv'

				if os.path.exists(path):
					df_trial = pd.read_csv(path)
					id_trial = list(df_trial['ID'])

					num_found = len(list(set(id_topk) & set(id_trial)))
					success_rate = 100.0 * num_found / num_search
					val_list.append(success_rate)
				else:
					val_list.append(0.0)
			contents.append(val_list)
		contents_list.append(contents)
	'''
	#mean_list = np.mean(contents_list, axis=1)
	#std_list = np.std(contents_list, axis=1)

	#print(mean_list)
	#print(std_list)

	plt.figure()
	#matplotlib.rcParams['text.usetex']=True
	print(contents)
	#contents=np.Array(contents)
	x_ = [1,2,3,4,5,6,7,8,9,10]
	plt.plot(x_, contents[0],'-o', c='yellowgreen', ms=5, label='Greedy')
	plt.plot(x_, contents[1],'-o', c='darkblue', ms=5, label='UCB')
	plt.plot(x_, contents[2],'-o', c='orangered', ms=5, label='Uncertainty')
	#plt.errorbar(x_, mean_list[0,:], yerr=std_list[0,:], c='b', ms=5, label='Greedy', marker='o', capsize=5)
	#plt.errorbar(x_, mean_list[0,:], yerr=std_list[0,:], c='b', ms=5, label='Greedy', marker='o', capsize=5)
	#plt.errorbar(x_, mean_list[1,:], yerr=std_list[1,:], c='g', ms=5, label='UCB', marker='o', capsize=5)
	#plt.errorbar(x_, mean_list[2,:], yerr=std_list[2,:], c='r', ms=5, label='Uncertainty', marker='o', capsize=5)

	plt.grid(True)
	plt.xlabel('i', fontsize=25)
	plt.ylabel('interval Hit Ratio (%)', fontsize=25)
	plt.xlim([0,10])
	plt.ylim([0,50])
	plt.xticks(ticks=[0, 5, 10],fontsize=25)
	plt.yticks(ticks=[0,10,20,30,40,50],fontsize=25)
	#plt.legend(fontsize=25)
	plt.tight_layout()
	plt.savefig('../figures/'+title+'_topk_vs_numsearch_HRi.png',dpi=600)


if __name__ == '__main__':
	main()
