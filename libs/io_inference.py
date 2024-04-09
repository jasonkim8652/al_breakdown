import os
import pandas as pd

import torch
import dgl
from dgl.data.utils import load_graphs

from rdkit import Chem


ATOM_VOCAB = [
	'C', 'N', 'O', 'S', 'F', 
	'H', 'Si', 'P', 'Cl', 'Br',
	'Li', 'Na', 'K', 'Mg', 'Ca',
	'Fe', 'As', 'Al', 'I', 'B',
	'V', 'Tl', 'Sb', 'Sn', 'Ag', 
	'Pd', 'Co', 'Se', 'Ti', 'Zn',
	'Ge', 'Cu', 'Au', 'Ni', 'Cd',
	'Mn', 'Cr', 'Pt', 'Hg', 'Pb'
]


def one_of_k_encoding(x, vocab):
	if x not in vocab:
		x = vocab[-1]
	return list(map(lambda s: float(x==s), vocab))


def get_atom_feature(atom):#making one hot encoding about the data
	atom_feature = one_of_k_encoding(atom.GetSymbol(), ATOM_VOCAB)
	atom_feature += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])#number of directly-bonded neighbor
	atom_feature += one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])#number of hydrogen
	atom_feature += one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])#number of implicit Hs on the atom
	atom_feature += [atom.GetIsAromatic()]
	return atom_feature
	

def get_bond_feature(bond):
	bt = bond.GetBondType()
	bond_feature = [
		bt == Chem.rdchem.BondType.SINGLE,
		bt == Chem.rdchem.BondType.DOUBLE,
		bt == Chem.rdchem.BondType.TRIPLE,
		bt == Chem.rdchem.BondType.AROMATIC,
		bond.GetIsConjugated(),
		bond.IsInRing()
	]
	return bond_feature


def get_molecular_graph(smi):
	mol = Chem.MolFromSmiles(smi)
	graph = dgl.DGLGraph()#just a class, it is constructor
	
	atom_list = mol.GetAtoms()
	num_atoms = len(atom_list)
	graph.add_nodes(num_atoms)

	atom_feature_list = [get_atom_feature(atom) for atom in atom_list]
	atom_feature_list = torch.tensor(atom_feature_list, dtype=torch.float64)
	graph.ndata['h'] = atom_feature_list

	bond_list = mol.GetBonds()
	bond_feature_list = []
	for bond in bond_list:
		bond_feature = get_bond_feature(bond)

		src = bond.GetBeginAtom().GetIdx()
		dst = bond.GetEndAtom().GetIdx()

		# DGL graph is undirectional
		# Thus, we have to add edge pair of both (i,j) and (j, i)
		# i --> j
		graph.add_edges(src, dst)
		bond_feature_list.append(bond_feature)

		# j --> i
		graph.add_edges(dst, src)
		bond_feature_list.append(bond_feature)
	
	bond_feature_list = torch.tensor(bond_feature_list, dtype=torch.float64)
	graph.edata['e_ij'] = bond_feature_list
	return graph


def smi_collate_fn(batch):
	graph_list = []
	for i, smi in enumerate(batch):
		graph = get_molecular_graph(smi)
		graph_list.append(graph)
	graph_list = dgl.batch(graph_list)
	return graph_list


def load_collate_fn(batch):
	graph_list = []
	for i, id_ in enumerate(batch):
		idx = int(id_.split('_')[1])
		conformer_idx = idx // 10000
		graph_path = os.path.join(
			'/home/jasonkjh/works/data/Enamine_HTS/',
                        'dgl_graph',
			str(conformer_idx),
			id_+'.bin'
		)
		if not os.path.exists(graph_path):
			print("NONE")
			continue
		graph = load_graphs(graph_path)
		graph_list.append(graph[0][0])
	graph_list = dgl.batch(graph_list)
	return graph_list


def attn_collate_fn(batch):
	graph_list = []
	id_list = []
	for i, batch in enumerate(batch):
		id_, smi = batch[0], batch[1]
		id_list.append(id_)
		smi_list.append(smi)

		idx = int(id_.split('_')[1])
		conformer_idx = idx // 10000
		graph_path = os.path.join(
			'/home/jasonkjh/works/Data/Enamine_HTS/',
			'dgl_graph',
			str(conformer_idx),
			id_+'.bin'
		)
		graph = load_graphs(graph_path)
		graph_list.append(graph[0][0])
	graph_list = dgl.batch(graph_list)
	return graph_list, id_list, smi_list


class SmiDataset(torch.utils.data.Dataset):
	def __init__(
			self, 
			smi_list
		):
		self.smi_list = smi_list
	
	def __len__(self):
		return len(self.smi_list)
	
	def __getitem__(
			self, 
			idx
		):
		return self.smi_list[idx]


class MyDataset(torch.utils.data.Dataset):
	def __init__(
			self, 
			id_list
		):
		self.id_list = id_list
	
	def __len__(self):
		return len(self.id_list)
	
	def __getitem__(
			self, 
			idx
		):
		return self.id_list[idx]


def debugging():
	print ("Hello!")


if __name__ == '__main__':
	debugging()
