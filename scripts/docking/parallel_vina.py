import os
import time
import argparse
import parmap
import pandas as pd
import multiprocessing as mp

from functools import partial


def run_single(
		inp,
		args,
	):
	id_, smi = inp[0], inp[1]
	#idx = int(id_.split('_')[1])
	#conformer_idx = idx // args.conformer_unit

	output_dir = os.path.join(
		args.current_dir,
		'outputs',
		args.title,
		#str(conformer_idx),
	)
	conformer_dir = os.path.join(
		args.conformer_dir,
		#str(conformer_idx),
	)

	ligand_pdbqt = os.path.join(
		args.conformer_dir,
		#str(conformer_idx),
                id_,
		id_ + '.pdbqt'
	)
	out_path = os.path.join(
		output_dir,
		id_+'.pdbqt'
	)
	if args.prepare_ligand:
		command = 'python /home/jasonkjh/works/active_learning/prepare_ligand.py'
		command += ' --smi \'' + smi + '\''
		command += ' --id_ ' + id_
		command += ' --dir_ ' + conformer_dir
		command += ' --pH ' + str(args.pH)
		os.system(command)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir,exist_ok=True)
	if os.path.exists(out_path):
		print("already_exists"+out_path)
		return
	command = 'vina '
	command += ' --receptor ' + args.receptor_pdbqt
	command += ' --ligand ' + ligand_pdbqt
	command += ' --config ' + args.config_path
	command += ' --out ' + out_path
	os.system(command)


def main(args):
	st = time.time()

	df = pd.read_csv(args.csv_path)
	'''
	if 'Dock' in df.columns:
		condition = (df['Dock'] == 0.0)
		df = df[condition] 
	idx_list = [i for i in range(args.start_idx, args.end_idx)]
	df = df.iloc[idx_list]
	'''
	num = len(df)
	print ("Total number of compounds to run docking:", num)
	print (df.head(50))

	id_list = ["Rand_"+str(i) for i in range(len(df))]
	smi_list = list(df['SMILES'])
	inp_list = list(zip(id_list, smi_list))

	num_cores = mp.cpu_count()
	fn_ = partial(
		run_single,
		args=args,
	)
	parmap.map(
		fn_,
		inp_list,
		pm_pbar=True,
		pm_processes=num_cores,
	)

	et = time.time()
	print ("Time for running:", round(et-st, 2), "(s)")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--title', type=str, required=True, 
						help='')
	parser.add_argument('--csv_path', type=str, required=True, 
						help='')
	parser.add_argument('--start_idx', type=int, required=True, 
						help='')
	parser.add_argument('--end_idx', type=int, required=True, 
						help='')

	parser.add_argument('--prepare_ligand', type=bool, default=False, 
						help='')

	parser.add_argument('--receptor_pdbqt', type=str, default='/home/jasonkjh/works/data/JAK2/JAK2_receptor.pdbqt', 
						help='')
	parser.add_argument('--config_path', type=str, default='/home/jasonkjh/works/projects/active_learning/config_JAK2.txt',
						help='')
	parser.add_argument('--conformer_dir', type=str, default='/home/jasonkjh/works/data/Real/Rand_1000', 
						help='')
	parser.add_argument('--current_dir', type=str, default='/home/jasonkjh/works/projects/active_learning/', 
						help='')

	parser.add_argument('--conformer_unit', type=int, default=10000, 
						help='')
	parser.add_argument('--pH', type=float, default=7.0,
						help='')

	args = parser.parse_args()

	main(args)
