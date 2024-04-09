import os
import time
import argparse
import parmap
import pandas as pd
import multiprocessing as mp

from functools import partial

from dgl.data.utils import save_graphs
from libs.io_inference import get_molecular_graph


def get_and_save_graph(
		inp,
		conformer_unit,
	):
	id_, smi = inp[0], inp[1]
	idx=int(id_.split("_")[1])
	conformer_idx = idx // conformer_unit
	#conformer_unit = id_.split("_")[1]
	dir_ = os.path.join(
		'/home/jasonkjh/works/data/Enamine_HTS/',
		'dgl_graph',
		str(conformer_idx)
	)
	if not os.path.exists(dir_):
		os.makedirs(dir_,exist_ok=True)

	graph = get_molecular_graph(smi)
	graph_path = os.path.join(
		dir_,
		id_+'.bin'
	)
	
	save_graphs(graph_path, [graph])


def main(args):
	st = time.time()
	df = pd.read_csv(args.csv_path)[221690:221700]
	id_list = list(df['ID'])
	smi_list= list(df['parent_SMILES'])

	inp_list = list(zip(id_list, smi_list))
	fn_ = partial(
		get_and_save_graph,
		conformer_unit=args.conformer_unit,
	)
	num_cores = mp.cpu_count()
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
	parser.add_argument('--csv_path', type=str, required=True, 
						help='')
	parser.add_argument('--conformer_unit', type=int, default=10000, 
						help='')
	args = parser.parse_args()

	main(args)
