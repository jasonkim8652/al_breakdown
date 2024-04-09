import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.backend import pytorch as F_dgl
from dgl.nn.functional import edge_softmax


class MLP(nn.Module):
	def __init__(
		self, 
		input_dim, 
		hidden_dim, 
		output_dim,
		bias=True,
		act=F.relu,
	):
		super().__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		self.act = act

		self.linear1 = nn.Linear(input_dim, hidden_dim, bias=bias)
		self.linear2 = nn.Linear(hidden_dim, output_dim, bias=bias)
	
	def forward(self, h):
		h = self.linear1(h)
		h = self.act(h)
		h = self.linear2(h)
		return h


class GraphConvolution(nn.Module):
	def __init__(
			self,
			hidden_dim,
			act=F.relu,
			dropout_prob=0.2,
		):
		super().__init__()

		self.act = act
		self.norm = nn.LayerNorm(hidden_dim)
		self.prob = dropout_prob
		self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
	
	def forward(
			self, 
			graph, 
			training=False
		):
		h0 = graph.ndata['h']

		graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'u_'))
		h = self.act(self.linear(graph.ndata['u_'])) + h0
		h = self.norm(h)
			
		# Apply dropout on node features
		h = F.dropout(h, p=self.prob, training=training)

		graph.ndata['h'] = h
		return graph


class GraphIsomorphism(nn.Module):
	def __init__(
			self,
			hidden_dim,
			act=F.relu,
			bias_mlp=True,
			dropout_prob=0.2,
		):
		super().__init__()

		self.mlp = MLP(
			input_dim=hidden_dim,
			hidden_dim=4*hidden_dim,
			output_dim=hidden_dim,
			bias=bias_mlp,
			act=act
		)
		self.norm = nn.LayerNorm(hidden_dim)
		self.prob = dropout_prob
	
	def forward(
			self, 
			graph, 
			training=False
		):
		h0 = graph.ndata['h']

		graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'u_'))
		h = self.mlp(graph.ndata['u_']) + h0
		h = self.norm(h)
			
		# Apply dropout on node features
		h = F.dropout(h, p=self.prob, training=training)

		graph.ndata['h'] = h
		return graph


class GraphIsomorphismEdge(nn.Module):
	def __init__(
			self,
			hidden_dim,
			act=F.relu,
			bias_mlp=True,
			dropout_prob=0.2,
		):
		super().__init__()

		self.norm = nn.LayerNorm(hidden_dim)
		self.prob = dropout_prob
		self.mlp = MLP(
			input_dim=hidden_dim,
			hidden_dim=4*hidden_dim,
			output_dim=hidden_dim,
			bias=bias_mlp,
			act=act,
		)
	
	def forward(
			self, 
			graph, 
			training=False
		):
		h0 = graph.ndata['h']

		graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
		graph.update_all(fn.copy_edge('e_ij', 'm_e'), fn.sum('m_e', 'u_'))
		u_ = graph.ndata['neigh'] + graph.ndata['u_']
		h = self.mlp(u_) + h0
		h = self.norm(h)
			
		# Apply dropout on node features
		h = F.dropout(h, p=self.prob, training=training)

		graph.ndata['h'] = h
		return graph


class GraphAttention(nn.Module):
	def __init__(
			self,
			hidden_dim,
			num_heads=4,
			bias_mlp=True,
			dropout_prob=0.2,
			act=F.relu,
		):
		super().__init__()

		self.mlp = MLP(
			input_dim=hidden_dim,
			hidden_dim=2*hidden_dim,
			output_dim=hidden_dim,
			bias=bias_mlp,
			act=act,
		)
		self.hidden_dim = hidden_dim
		self.num_heads = num_heads
		self.splitted_dim = hidden_dim // num_heads

		self.prob = dropout_prob

		self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w4 = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w5 = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w6 = nn.Linear(hidden_dim, hidden_dim, bias=False)

		self.act = F.elu
		self.norm = nn.LayerNorm(hidden_dim)
	
	def forward(
			self, 
			graph, 
			training=False
		):
		h0 = graph.ndata['h']
		e_ij = graph.edata['e_ij']

		graph.ndata['u'] = self.w1(h0).view(-1, self.num_heads, self.splitted_dim)
		graph.ndata['v'] = self.w2(h0).view(-1, self.num_heads, self.splitted_dim)
		graph.edata['x_ij'] = self.w3(e_ij).view(-1, self.num_heads, self.splitted_dim)

		graph.apply_edges(fn.v_add_e('v', 'x_ij', 'm'))
		graph.apply_edges(fn.u_mul_e('u', 'm', 'attn'))
		graph.edata['attn'] = edge_softmax(graph, graph.edata['attn'] / math.sqrt(self.splitted_dim))
	

		graph.ndata['k'] = self.w4(h0).view(-1, self.num_heads, self.splitted_dim)
		graph.edata['x_ij'] = self.w5(e_ij).view(-1, self.num_heads, self.splitted_dim)
		graph.apply_edges(fn.v_add_e('k', 'x_ij', 'm'))

		graph.edata['m'] = graph.edata['attn'] * graph.edata['m']
		graph.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h'))
		
		h = self.w6(h0) + graph.ndata['h'].view(-1, self.hidden_dim)
		h = self.norm(h)

		# Add and Norm module
		h = h + self.mlp(h)
		h = self.norm(h)
			
		# Apply dropout on node features
		h = F.dropout(h, p=self.prob, training=training)

		graph.ndata['h'] = h 
		return graph


class PMALayer(nn.Module):
	def __init__(
			self,
			k,
			hidden_dim,
			num_heads,
			multiply_num_pma=False,
		):
		super().__init__()

		self.k = k 
		self.hidden_dim = hidden_dim,
		self.num_heads = num_heads

		self.mha = MultiHeadAttention(
			hidden_dim=hidden_dim,
			num_heads=num_heads,
			multiply_num_pma=multiply_num_pma,
		)
		self.seed_vec = torch.ones(1)
		self.w_seed = nn.Linear(1, k*hidden_dim)

	def forward(
			self,
			graph,
		):
		h = graph.ndata['h']

		lengths = graph.batch_num_nodes()
		batch_size = len(lengths)
		
		device = h.device
		self.seed_vec = self.seed_vec.to(device)
		if self.k == 1:
			query = self.w_seed(self.seed_vec).repeat(batch_size, 1)
		else:
			query = self.w_seed(self.seed_vec).reshape(self.k, -1).repeat(batch_size, 1)

		out, alpha = self.mha(
			query,
			h,
			[self.k] * batch_size,
			lengths,
		)
		return out, alpha


class MultiHeadAttention(nn.Module):
	def __init__(
			self,
			hidden_dim,
			num_heads,
			multiply_num_pma=False,
		):
		super().__init__()

		self.hidden_dim = hidden_dim
		self.num_heads = num_heads
		self.d_heads = hidden_dim // num_heads
		self.multiply_num_pma = multiply_num_pma

		self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.w_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
	

	def forward(
			self,
			q,
			v,
			lengths_q,
			lengths_v,
		):
		batch_size = len(lengths_q)
		max_len_q = max(lengths_q)
		max_len_v = max(lengths_v)

		q = self.w_q(q).view(-1, self.num_heads, self.d_heads)
		k = self.w_k(v).view(-1, self.num_heads, self.d_heads)
		v = self.w_v(v).view(-1, self.num_heads, self.d_heads)

		q = F_dgl.pad_packed_tensor(q, lengths_q, 0)
		k = F_dgl.pad_packed_tensor(k, lengths_v, 0)
		v = F_dgl.pad_packed_tensor(v, lengths_v, 0)

		#e = torch.einsum('bxhd,byhd->bhxy', q, k)
		q = q.tile(1, max_len_v, 1, 1)
		e = q*v
		e = torch.sum(e, -1)
		e = e.permute(0, 2, 1)
		e = e.unsqueeze(2)
		e = e / math.sqrt(self.d_heads)

		mask = torch.zeros(batch_size, max_len_q, max_len_v).to(e.device)
		for i in range(batch_size):
			mask[i, :lengths_q[i], :lengths_v[i]].fill_(1)
		mask = mask.unsqueeze(1)
		e.masked_fill_(mask == 0, -float('inf'))

		alpha = torch.softmax(e, dim=-1)

		#out = torch.einsum('bhxy,bhyd->bxhd', alpha, v)
		alpha = alpha.permute(0,1,3,2)
		v = v.permute(0,2,1,3)
		out = alpha * v
		out = torch.sum(out, 2)
		out = out.unsqueeze(1)

		out = self.w_o(
			out.contiguous().view(batch_size, max_len_q, self.hidden_dim)
		)
		out = F_dgl.pack_padded_tensor(out, lengths_q)

		if self.multiply_num_pma:
			out = out * torch.tensor(lengths_v).unsqueeze(-1).repeat(1, self.hidden_dim)
		
		return out, alpha
