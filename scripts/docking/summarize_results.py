import os
import time
import argparse
import parmap
import multiprocessing as mp

import pandas as pd
import numpy as np

from functools import partial


def get_docking_score(
		id_,
		title,
	):
	#idx = int(id_.split('_')[1])
	#conformer_idx = idx // args.conformer_unit
	out_path = os.path.join(
		'/home/jasonkjh/works/projects/active_learning',
		'outputs',
		title,
		#str(conformer_idx),
		id_+'.pdbqt'
	)

	score = 0.0
	try:
		if os.path.exists(out_path):
			f = open(out_path)
			lines = f.readlines()
			if len(lines) > 0:
				score = float(lines[1].split()[3])
	except:
		score = 0.0
	return score


def report(df):
	num_total = len(df)
	condition = (df['Dock'] < 0.0)
	df = df[condition]
	score_list = list(df['Dock'])
	print ("Number of total calculations:", num_total)
	print ("Number of finished calculations:", len(score_list))
	print ("Number of unfinished calculations:", num_total - len(score_list))
	print ("Max: ", max(score_list), "Min: ", min(score_list))


def main(args):
	st = time.time()
	cwd = os.getcwd()

	df_hts = pd.read_csv(args.csv_path)
	id_list = list(df_hts['ID'])
	smi_list = list(df_hts['SMILES'])

	num_cores = mp.cpu_count()
	fn_ = partial(
		get_docking_score,
		title=args.title,
	)
	score_list = parmap.map(
		fn_,
		id_list,
		pm_pbar=True,
		pm_processes=num_cores,
	)
        #score_list = fn_(id_list)
	et = time.time()
	print ("Time for running:", round(et-st, 2), "(s)")
	
	df_new = pd.DataFrame({})
	df_new['ID'] = id_list
	df_new['SMILES'] = smi_list
	df_new['Dock'] = score_list

	df_new.to_csv('./results/'+args.title+'_dock.csv', index=False)

	report(df=df_new)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--title', type=str, required=True, 
						help='')
	parser.add_argument('--csv_path', type=str, required=True, 
						help='')
	parser.add_argument('--conformer_unit', type=int, default=10000, 
						help='')
	args = parser.parse_args()

	main(args)

