#!/usr/bin/env python3

import os
import time
import argparse
import sys
from functools import partial
import parmap
import multiprocessing as mp

import numpy as np
import pandas as pd
sys.path.add("../../../")
from libs.plip_modules import extract_interaction
from libs.plip_modules import prepare_complex
from libs.plip_modules import flush

from plip.basic import config

#config.HYDROPH_DIST_MAX = 1.0


def run_plip(
		id_,
		task_type,
		receptor_path,
		docking_dir,
		ligand_id='UNK',
	):
	
	job_dir = os.path.join(docking_dir)
	idx = str(int(id_.split('_')[1])//10000)
	ligand_path = os.path.join(job_dir,idx, id_+'.pdbqt')
	if os.path.exists(ligand_path):
		#flush(id_, job_dir)
		prepare_complex(
			id_,
			job_dir,
			ligand_path=ligand_path,
			receptor_path=receptor_path,
		)
		complex_pdb = os.path.join(job_dir, id_+'_complex.pdb')
		try:
			pi_profile = extract_interaction(
				complex_pdb=complex_pdb,
				ligand_id=ligand_id
			)
		except:
			return [0 for _ in range(5)] + ['' for _ in range(5)]
		#flush(id_, job_dir)
		return pi_profile

	else:
		zero_profile = [0 for _ in range(5)] + ['' for _ in range(5)]
		return zero_profile


def main(args):
	st = time.time()
	dir_ = os.path.abspath(args.dir_)
	#idx_ = int(args.idx)
	docking_dir = "/home/jasonkjh/works/projects/active_learning/outputs/EGFR"
	receptor_path = os.path.abspath(args.receptor_path)

	df = pd.read_csv(args.input_csv)
	#df = df[idx_*int(len(df)/10):(idx_+1)*int(len(df)/10)]
	condition = (df['Dock'] < 0.0)
	df = df[condition]
	print ("Number of molecules to analyze:", len(df))

	id_list = list(df['ID'])
	
	fn_ = partial(
		run_plip,
		task_type=args.task_type,
		receptor_path=receptor_path,
		docking_dir=docking_dir,
		ligand_id=args.ligand_id,
	)
	
	profile_list = parmap.map(
		fn_,
		id_list,
		pm_pbar=True,
		pm_processes=mp.cpu_count(),
	)
	'''
	profile_list = []
	for i,id_ in enumerate(id_list):
		profile_ = run_plip(
			id_=id_,
			task_type=args.task_type,
			receptor_path=receptor_path,
			docking_dir=docking_dir,
			ligand_id='UNK',
		)
		#print(profile_)
		profile_list.append(profile_)
		if i%100 == 0:
			print (i, "th molecule is analyzed")
		#print (id_)
	'''
	profile_list = np.asarray(profile_list)
	#print(profile_list)
	df['Num_HBonds'] = list(profile_list[:,0].astype(int))
	df['Num_Pication'] = list(profile_list[:,1].astype(int))
	df['Num_Pistack'] = list(profile_list[:,2].astype(int))
	df['Num_Hydrophobic'] = list(profile_list[:,3].astype(int))
	df['Num_Halogen'] = list(profile_list[:,4].astype(int))
	df['Res_HBonds'] = list(profile_list[:,5])
	df['Res_Pication'] = list(profile_list[:,6])
	df['Res_Pistack'] = list(profile_list[:,7])
	df['Res_Hydrophobic'] = list(profile_list[:,8])
	df['Res_Halogen'] = list(profile_list[:,9])
	print (df)

	csv_path = os.path.join(
		args.prefix + '_plip_analysis_.csv'
	)
	df.to_csv('/home/jasonkjh/works/data/EGFR/'+csv_path, index=False)
	et = time.time()
	print ("Time for plip-analysis:", round(et-st, 2), "(s)")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_csv', type=str, required=True, 
						help='Input csv, absolute path recommended')
	parser.add_argument('-r', '--receptor_path', type=str, required=True, 
						help='Receptor PDB file')
	parser.add_argument('-d', '--dir_', type=str, default="/home/jasonkjh/tmp", 
						help='Directory to save the files')
	parser.add_argument('-t', '--task_type', type=str, default='gd2', 
						help='Options: gd2, gd3, gd2_align, gd3_align')
	parser.add_argument('-p', '--prefix', type=str, required=True, 
						help='prefix of the final csv and mol2 files')
	parser.add_argument('-l', '--ligand_id', type=str, default='UNK', 
						help='ID of ligand')
	
	
	args = parser.parse_args()
	main(args)
