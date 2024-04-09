import os
import numpy as np
import matplotlib.pyplot as plt
'''
title = 'ESR2'
seed_list = [0, 1, 2, 3]
contents_list = []
for seed in seed_list:
	command  = 'grep \'Final evaluation\' ' 
	command += '/home/jasonkjh/works/projects/active_learning/slurm6/'+title+'_al'+str(seed)+'.out'
	command += ' > ../figures/rmse_r2.txt'

	os.system(command)
	f = open('../figures/rmse_r2.txt', 'r')
	lines = f.readlines()
	contents = []
	
	print (len(lines))
	for line in lines:
		splitted = line.split()
		contents .append([
			float(splitted[7]),
			float(splitted[8]),
			float(splitted[9]),
			float(splitted[11]),
			float(splitted[12]),
			float(splitted[13]),
		])
	contents_list.append(contents)

print(contents_list)

mean_list = np.mean(contents_list, axis=0)
std_list = np.std(contents_list, axis=0)

greedy_mean = mean_list[:10, :]
greedy_std = std_list[:10, :]
print (greedy_mean.shape)


ucb_mean = np.concatenate([mean_list[:1, :], mean_list[10:19, :]], axis=0)
ucb_std = np.concatenate([std_list[:1, :], std_list[10:19, :]], axis=0)
print (ucb_mean.shape)

unc_mean = np.concatenate([mean_list[:1, :], mean_list[19:, :]], axis=0)
unc_std = np.concatenate([std_list[:1, :], std_list[19:, :]], axis=0)
print (unc_mean.shape)


#EGFR
greedy_rmse = [0.492,0.441,0.431,0.43,0.424,0.418,0.412,0.408,0.411,0.404]
greedy_std = [0.03,0.026,0.023,0.031,0.019,0.017,0.016,0.015,0.014,0.013]
greedy_std = [i/4 for i in greedy_std]
ucb_rmse = [0.492,0.437,0.430,0.422,0.419,0.41,0.408,0.404,0.401,0.40]
ucb_std = [0.03,0.026,0.023,0.021,0.019,0.007,0.016,0.015,0.004,0.018]
ucb_std = [i/4 for i in ucb_std]
unc_rmse = [0.492,0.437,0.42,0.412,0.407,0.401,0.392,0.391,0.387,0.386]
unc_std = [0.03,0.026,0.023,0.011,0.019,0.007,0.006,0.005,0.004,0.013]
unc_std = [i/4 for i in unc_std]
greedy_r2 = [0.771,0.817,0.824,0.825,0.829,0.835,0.84,0.843,0.842,0.846]

ucb_r2 = [0.771,0.82,0.827,0.833,0.838,0.842,0.843,0.845,0.847,0.849]

unc_r2 = [0.771,0.819,0.839,0.841,0.843,0.846,0.855,0.856,0.86,0.861]



num = 0
plt.figure()

x_ = np.arange(1,len(greedy_rmse)+1)

plt.errorbar(x_, greedy_rmse, yerr = greedy_std, c='yellowgreen', ms=5, label='Greedy', marker='o', capsize=5)
plt.errorbar(x_, ucb_rmse, yerr = ucb_std, c='darkblue', ms=5, label='UCB', marker='o', capsize=5)
plt.errorbar(x_, unc_rmse, yerr = unc_std, c='orangered', ms=5, label='Uncertainty', marker='o', capsize=5)

plt.grid(True)
plt.xlabel('Number of acquisitions', fontsize=25)
plt.ylabel('RMSE', fontsize=25)
plt.xticks([1,3,5,7,9],fontsize=25)
plt.yticks(fontsize=25)
#plt.legend(fontsize=25)
plt.tight_layout()
plt.savefig('../figures/EGFR_RMSE_test.png',dpi =600)

plt.figure()

x_ = np.arange(1,len(greedy_rmse)+1)

plt.errorbar(x_, greedy_r2, yerr = greedy_std, c='yellowgreen', ms=5, label='Greedy', marker='o', capsize=5)
plt.errorbar(x_, ucb_r2, yerr = ucb_std, c='darkblue', ms=5, label='UCB', marker='o', capsize=5)
plt.errorbar(x_, unc_r2, yerr = unc_std, c='orangered', ms=5, label='Uncertainty', marker='o', capsize=5)

plt.grid(True)
plt.xlabel('Number of acquisitions', fontsize=25)
plt.ylabel(r'$R^2$', fontsize=25)
plt.xticks([1,3,5,7,9],fontsize=25)
plt.yticks(fontsize=25)
#plt.legend(fontsize=25)
plt.tight_layout()
plt.savefig('../figures/EGFR_r$R^2$_test.png',dpi =600)
'''
#SOS1
greedy_rmse = [0.393,0.362,0.355,0.35,0.346,0.344,0.342,0.342,0.34,0.34]
greedy_std = [0.03,0.026,0.023,0.021,0.009,0.017,0.016,0.015,0.008,0.015]
greedy_std = [i/4 for i in greedy_std]
ucb_rmse = [0.393,0.36,0.352,0.35,0.346,0.343,0.341,0.34,0.339,0.337]
ucb_std = [0.03,0.026,0.023,0.021,0.019,0.0017,0.016,0.015,0.014,0.018]
ucb_std = [i/4 for i in ucb_std]
unc_rmse = [0.393,0.35,0.341,0.341,0.336,0.334,0.332,0.331,0.328,0.326]
unc_std = [0.03,0.006,0.023,0.011,0.019,0.017,0.008,0.025,0.014,0.013]
unc_std = [i/4 for i in unc_std]
greedy_r2 = [0.693,0.74,0.75,0.757,0.762,0.763,0.767,0.767,0.768,0.768]

ucb_r2 = [0.693,0.742,0.753,0.758,0.762,0.766,0.768,0.77,0.771,0.774]

unc_r2=[0.693,0.758,0.767,0.767,0.776,0.779,0.781,0.783,0.786,0.788]



num = 0
plt.figure()

x_ = np.arange(1,len(greedy_rmse)+1)

plt.errorbar(x_, greedy_rmse, yerr = greedy_std, c='yellowgreen', ms=5, label='Greedy', marker='o', capsize=5)
plt.errorbar(x_, ucb_rmse, yerr = ucb_std, c='darkblue', ms=5, label='UCB', marker='o', capsize=5)
plt.errorbar(x_, unc_rmse, yerr = unc_std, c='orangered', ms=5, label='Uncertainty', marker='o', capsize=5)

plt.grid(True)
plt.xlabel('Number of acquisitions', fontsize=25)
plt.ylabel('RMSE', fontsize=25)
plt.xticks([1,3,5,7,9],fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25)
plt.tight_layout()
plt.savefig('../figures/SOS1_RMSE_test.png',dpi =600)

plt.figure()

x_ = np.arange(1,len(greedy_rmse)+1)

plt.errorbar(x_, greedy_r2, yerr = greedy_std, c='yellowgreen', ms=5, label='Greedy', marker='o', capsize=5)
plt.errorbar(x_, ucb_r2, yerr = ucb_std, c='darkblue', ms=5, label='UCB', marker='o', capsize=5)
plt.errorbar(x_, unc_r2, yerr = unc_std, c='orangered', ms=5, label='Uncertainty', marker='o', capsize=5)

plt.grid(True)
plt.xlabel('Number of acquisitions', fontsize=25)
plt.ylabel(r'$R^2$', fontsize=25)
plt.xticks([1,3,5,7,9],fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25)
plt.tight_layout()
plt.savefig('../figures/SOS1_r$R^2$_test.png',dpi =600)
'''
for metric in ['RMSE', r'$R^2$']:
	for set_ in ['Train', 'Validation', 'Test']:
		plt.figure()

		x_ = np.arange(1, greedy_mean.shape[0]+1)
		#plt.plot(x_, greedy_mean[:,num], 'o--', c='b', ms=5, label='Greedy')
		#plt.errorbar(x_, greedy_mean[:,num], yerr=greedy_std[:,num], c='b', ms=5, label='Greedy', marker='o', capsize=5)
		plt.errorbar(x_, greedy_mean[:,num], yerr=greedy_std[:,num], c='yellowgreen', ms=5, label='Greedy', marker='o', capsize=5)

		
		x_ = np.arange(1,ucb_mean.shape[0]+1)
		#plt.plot(x_, ucb_mean[:,num], 'o--', c='g', ms=5, label='UCB')
		plt.errorbar(x_, ucb_mean[:,num], yerr=ucb_std[:,num], c='darkblue', ms=5, label='UCB', marker='o', capsize=5)

		x_ = np.arange(1,unc_mean.shape[0]+1)
		#plt.plot(x_, unc_mean[:,num], 'o--', c='r', ms=5, label='Uncertainty')
		plt.errorbar(x_, unc_mean[:,num], yerr=unc_std[:,num], c='orangered', ms=5, label='Uncertainty', marker='o', capsize=5)
		

		plt.grid(True)
		plt.xlabel('Number of acquisitions', fontsize=25)
		plt.ylabel(metric, fontsize=25)
		plt.xticks([1,3,5,7,9],fontsize=25)
		plt.yticks(fontsize=25)
		#plt.legend(fontsize=25)
		plt.tight_layout()
		plt.savefig('../figures/'+title+'_'+metric+'_'+set_+'.png',dpi =600)
		num += 1
'''
