import os
import time
import argparse

from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from libs.io_inference import SmiDataset
from libs.io_inference import smi_collate_fn as collate_fn

from libs.models import MyModel

from libs.utils import str2bool
from libs.utils import set_seed
from libs.utils import set_device


def main(args):
	# Set random seeds and device
	set_seed(seed=1234)
	device = torch.device('cuda')

	df = pd.read_csv(args.csv_path)
	smi_list = list(df['SMILES'])
	test_ds = SmiDataset(smi_list=smi_list)
	test_loader = DataLoader(
		dataset=test_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=collate_fn
	)

	# Construct model and load trained parameters if it is possible
	model = MyModel(
		model_type=args.model_type,
		num_layers=args.num_layers,
		hidden_dim=args.hidden_dim,
		readout=args.readout,
		dropout_prob=args.dropout_prob,
		out_dim=args.out_dim,
		multiply_num_pma=args.multiply_num_pma,
	)
	model = model.to(device)

	save_path = '/home/jasonkjh/works/active_learning/save/' 
	save_path += str(args.title)
	save_path += '_' + str(args.seed)
	save_path += '_' + str(args.step)
	if args.step != 0:
		save_path += '_' + str(args.method)
	save_path += '.pth'
	ckpt = torch.load(save_path, map_location=device)
	model.load_state_dict(ckpt['model_state_dict'])

	print ("Hello")
	model.eval()
	with torch.no_grad():
		# Test
		pred_list = []
		ale_unc_list = []
		epi_unc_list = []
		for i, graph in enumerate(test_loader):
			st = time.time()
			graph = graph.to(device)
			feat=graph.ndata['h']			
			pred, alpha = model(graph, feat, training=False)

			pred_list.append(pred[:,0])
			ale_unc_list.append(torch.exp(pred[:,1]))
			print (i, "/", len(test_loader))

	pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()
	ale_unc_list = torch.cat(ale_unc_list, dim=0).detach().cpu().numpy()
	ale_unc_list = np.sqrt(ale_unc_list)

	pred_list = list(np.around(pred_list, 3))
	ale_unc_list = list(np.around(ale_unc_list, 3))

	df['Pred'] = pred_list
	df['Unc'] = ale_unc_list

	df.to_csv(args.new_path, index=False)

	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv_path', type=str, required=True, 
						help='')
	parser.add_argument('--new_path', type=str, required=True, 
						help='')
	parser.add_argument('--title', type=str, required=True, 
						help='Job title of this execution')
	parser.add_argument('--seed', type=int, required=True, 
						help='Seed used for dataset splitting')
	parser.add_argument('--step', type=int, required=True, 
						help='Step number in active learning')
	parser.add_argument('--method', type=str, required=True, 
						help='Method used for active learning')

	parser.add_argument('--use_gpu', type=str2bool, default=True, 
						help='whether to use GPU device')
	parser.add_argument('--gpu_idx', type=str, default='1', 
						help='index of gpu to use')

	parser.add_argument('--model_type', type=str, default='gine', 
						help='Type of GNN model, Options: gcn, gin, gin_e, gat, ggnn')
	parser.add_argument('--num_layers', type=int, default=4,
						help='Number of GIN layers for ligand featurization')
	parser.add_argument('--hidden_dim', type=int, default=128,
						help='Dimension of hidden features')
	parser.add_argument('--out_dim', type=int, default=2,
						help='Dimension of final outputs')
	parser.add_argument('--readout', type=str, default='pma', 
						help='Readout method, Options: sum, mean, ...')
	parser.add_argument('--dropout_prob', type=float, default=0.2, 
						help='Probability of dropout on node features')
	parser.add_argument('--multiply_num_pma', type=str2bool, default=False, 
						help='whether to multiply number of atoms in the PMA layer')

	parser.add_argument('--num_workers', type=int, default=8,
						help='Number of workers to run dataloaders')
	parser.add_argument('--batch_size', type=int, default=100,
						help='Number of samples in a single batch')

	args = parser.parse_args()

	print ("Arguments")
	for k, v in vars(args).items():
		print (k, ": ", v)
	main(args)
