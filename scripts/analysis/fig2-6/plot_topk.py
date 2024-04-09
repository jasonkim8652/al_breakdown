import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def main():
    seed_list = [i for i in range(4)]
    step_list = [i for i in range(10)]
    method_list = ['greedy', 'ucb','unc']
    #method_list = ['greedy','ucb']
    title = 'PGR'
    num_search = 1000
    #df_all = pd.read_csv('/home/seongok/works/active_learning/results/6SCM_Enamine_final.csv')
    df_all = pd.read_csv('/home/jasonkjh/works/data/'+title+'/'+title+'.csv')
    df_topk = df_all.sort_values(by=['Dock'], ascending=True).head(num_search)
    id_topk = list(df_topk['ID'])
    print (len(id_topk))

    contents_list = []
    for method in method_list:
        contents = []
        for seed in seed_list:
            val_list = []
            for step in step_list:
                path = '/home/jasonkjh/works/projects/active_learning/data/'
                path += title 
                path += '_seed' + str(seed)
                path += '_step' + str(step)
                if step != 0:
                    path += '_' + method 
                path += '_train.csv'

                if os.path.exists(path):
                    print(path)
                    df_trial = pd.read_csv(path)
                    id_trial = list(df_trial['ID'])

                    num_found = len(list(set(id_topk) & set(id_trial)))
                    success_rate = 100.0 * num_found / num_search
                    val_list.append(success_rate)
                else:
                    val_list.append(0.0)
            contents.append(val_list)
        contents_list.append(contents)
    
    mean_list = np.mean(contents_list, axis=1)
    std_list = np.std(contents_list, axis=1)
    
    #mean_list = np.append(mean_list,np.array([[0.0,20.2,33.1,38.4,39.8,42.2,46.3,51.2,55.3,58.3]]),axis=0)
    #std_list = np.append(std_list,np.array([[0.0,5.2,6.7,7.5,6.3,6.2,5.1,4.4,4.2,4.5]]),axis=0)
    print(mean_list)
    print(std_list)
	
    plt.figure()
    x_ = np.arange(1,mean_list.shape[1]+1)
    #plt.plot(x_, mean_list[0,:], 'o--', c='b', ms=5, label='Greedy')
    #plt.plot(x_, mean_list[1,:], 'o--', c='g', ms=5, label='UCB')
    #plt.plot(x_, mean_list[2,:], 'o--', c='r', ms=5, label='Uncertainty')
    plt.errorbar(x_, mean_list[0,:], yerr=std_list[0,:], c='yellowgreen', ms=5, label='Greedy', marker='o', capsize=5)
    #plt.errorbar(x_, mean_list[0,:], yerr=std_list[0,:], c='b', ms=5, label='Greedy', marker='o', capsize=5)
    plt.errorbar(x_, mean_list[1,:], yerr=std_list[1,:], c='darkblue', ms=5, label='UCB', marker='o', capsize=5)
    plt.errorbar(x_, mean_list[2,:], yerr=std_list[2,:], c='orangered', ms=5, label='Uncertainty', marker='o', capsize=5)

    plt.grid(True)
    plt.xlabel('Number of acquisitions', fontsize=25)
    plt.ylabel('Top-' + str(num_search) + ' molecules found (%)', fontsize=25)
    plt.xticks([1,3,5,7,9],fontsize=25)
    plt.yticks(fontsize=25)
    #plt.legend(fontsize=25)
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.savefig('../figures/'+title+'_top'+str(num_search)+'_success.png',dpi = 600)


if __name__ == '__main__':
	main()
