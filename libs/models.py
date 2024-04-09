import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl

from libs.layers import GraphConvolution
from libs.layers import GraphIsomorphism
from libs.layers import GraphIsomorphismEdge
from libs.layers import GraphAttention
from libs.layers import PMALayer


class MyModel(nn.Module):
	def __init__(
			self, 
			model_type,
			num_layers=4, 
			hidden_dim=64,
			num_heads=4, # Only used for GAT
			dropout_prob=0.2,
			bias_mlp=True,
			out_dim=1,
			readout='sum',
			act=F.relu,
			initial_node_dim=58,
			initial_edge_dim=6,
			apply_sigmoid=False,
			multiply_num_pma=False,
		):
		super().__init__()

		self.num_layers = num_layers
		self.embedding_node = nn.Linear(initial_node_dim, hidden_dim, bias=False)
		self.embedding_edge = nn.Linear(initial_edge_dim, hidden_dim, bias=False)
		self.readout = readout

		self.mp_layers = torch.nn.ModuleList()
		for _ in range(self.num_layers):
			mp_layer = None
			if model_type == 'gcn':
				mp_layer = GraphConvolution(
					hidden_dim=hidden_dim,
					dropout_prob=dropout_prob,
					act=act,
				)
			elif model_type == 'gin':
				mp_layer = GraphIsomorphism(
					hidden_dim=hidden_dim,
					dropout_prob=dropout_prob,
					act=act,
					bias_mlp=bias_mlp
				)
			elif model_type == 'gine':
				mp_layer = GraphIsomorphismEdge(
					hidden_dim=hidden_dim,
					dropout_prob=dropout_prob,
					act=act,
					bias_mlp=bias_mlp
				)
			elif model_type == 'gat':
				mp_layer = GraphAttention(
					hidden_dim=hidden_dim,
					num_heads=num_heads,
					dropout_prob=dropout_prob,
					act=act,
					bias_mlp=bias_mlp
				)
			else:
				raise ValueError('Invalid model type: you should choose model type in [gcn, gin, gin, gat, ggnn]')
			self.mp_layers.append(mp_layer)

		if self.readout == 'pma':
			self.pma = PMALayer(
				k=1,
				hidden_dim=hidden_dim,
				num_heads=num_heads,
				multiply_num_pma=multiply_num_pma
			)

		self.linear_out = nn.Linear(hidden_dim, out_dim, bias=True)

		self.apply_sigmoid = apply_sigmoid
		if self.apply_sigmoid:
			self.sigmoid = F.sigmoid
	

	def forward(
			self, 
			graph,
			feat,
			eweight=None,
			training=False,
		):
		h = self.embedding_node(feat.float())
		if eweight==None:
			eweight=graph.edata['e_ij']
		e_ij = self.embedding_edge(eweight.float())
		graph.ndata['h'] = h
		graph.edata['e_ij'] = e_ij

# Update the node features
		for i in range(self.num_layers):
			graph = self.mp_layers[i](
				graph=graph,
				training=training
				)

# Aggregate the node features and apply the last linear layer to compute the logit
		alpha = None
		if self.readout in ['sum', 'mean', 'max']:
			out = dgl.readout_nodes(graph, 'h', op=self.readout)
		elif self.readout == 'pma':
			out, alpha = self.pma(graph)
		out = self.linear_out(out)

		if self.apply_sigmoid:
			out = self.sigmoid(out)

		return out, alpha


class MLP_model(nn.Module):
	def __init__(
			self, 
			num_layers=3, 
			inp_dim=1024,
			hidden_dim=512,
			dropout_prob=0.2,
			out_dim=1,
			act=F.relu,
			apply_sigmoid=False,
		):
		super().__init__()

		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.dropout_prob = dropout_prob
		self.apply_sigmoid = apply_sigmoid

		self.linear1 = nn.Linear(inp_dim, hidden_dim, bias=True)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
		self.linear3 = nn.Linear(hidden_dim, out_dim, bias=True)
		self.act = act

		if self.apply_sigmoid:
			self.sigmoid = F.sigmoid
	
	def forward(self, x, training=False):
		x = x.float()

		out = self.linear1(x)
		out = self.act(out)
		out = F.dropout(out, p=self.dropout_prob, training=training)
		out = self.linear2(out)
		out = self.act(out)
		out = F.dropout(out, p=self.dropout_prob, training=training)
		out = self.linear3(out)

		if self.apply_sigmoid:
			out = self.sigmoid(out)
		return out

