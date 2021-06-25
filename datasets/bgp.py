import json
import shutil

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import *


def convert_ndarray(x):
	y = list(range(len(x)))
	for k, v in x.items():
		y[int(k)] = v
	return np.array(y)


def check_rm(neighbors_set, unlabeled_nodes):
	for node in neighbors_set:
		if node not in unlabeled_nodes:
			return False
	return True


def rm_useless(G, feats, class_map, unlabeled_nodes, num_layers):
	# find useless nodes
	print('start to check and remove {} unlabeled nodes'.format(len(unlabeled_nodes)))

	rm_nodes = unlabeled_nodes
	if len(rm_nodes):
		for node in rm_nodes:
			G.remove_node(node)
		G_new = nx.relabel.convert_node_labels_to_integers(G, ordering='sorted')
		feats = np.delete(feats, rm_nodes, 0)
		class_map = np.delete(class_map, rm_nodes, 0)
		print('remove {} '.format(len(rm_nodes)), 'useless unlabeled nodes')
	return G_new, feats, class_map

class BGP(InMemoryDataset):


	def __init__(self, root, transform=None, pre_transform=None):

		self.dump_location = "raw_data_src/bgp_data_dump"
		super(BGP, self).__init__(root, transform, pre_transform)
		self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def raw_file_names(self):
		return ['as-G.json', 'as-class_map.json', 'as-feats.npy','as-feats_t.npy', 'as-edge_list']

	@property
	def processed_file_names(self):
		return 'data.pt'

	def download(self):
		for name in self.raw_file_names:
			# download_url('{}/{}'.format(self.url, name), self.raw_dir)
			source = self.dump_location + '/' + name
			shutil.copy(source, self.raw_dir)

	def process(self):
		G = nx.json_graph.node_link_graph(json.load(open(self.raw_paths[0])), False)
		class_map = json.load(open(self.raw_paths[1]))
		feats = np.load(self.raw_paths[2])
		feats_t = np.load(self.raw_paths[3])


		train_nodes = [n for n in G.nodes() if not G.nodes[n]['test'] and not G.nodes[n]['val']]
		val_nodes = [n for n in G.nodes() if not G.nodes[n]['test'] and G.nodes[n]['val']]
		test_nodes = [n for n in G.nodes() if G.nodes[n]['test'] and not G.nodes[n]['val']]
		unlabeled_nodes = [n for n in G.nodes() if G.nodes[n]['test'] and G.nodes[n]['val']]
		class_map = convert_ndarray(class_map)

		G, feats, class_map = rm_useless(G, feats, class_map, unlabeled_nodes, 1)
		train_nodes = [n for n in G.nodes() if not G.nodes[n]['test'] and not G.nodes[n]['val']]
		val_nodes = [n for n in G.nodes() if not G.nodes[n]['test'] and G.nodes[n]['val']]
		test_nodes = [n for n in G.nodes() if G.nodes[n]['test'] and not G.nodes[n]['val']]
		unlabeled_nodes = [n for n in G.nodes() if G.nodes[n]['test'] and G.nodes[n]['val']]


		data = from_networkx(G)
		data.train_mask = ~(data.test | data.val)
		data.val_mask = data.val
		data.test_mask = data.test
		data.test = None
		data.val = None
		data.x = torch.FloatTensor(feats)
		data.y = torch.LongTensor(np.argmax(class_map, axis=1))

		if self.pre_filter is not None:
			data = self.pre_filter(data)

		if self.pre_transform is not None:
			data.edge_index = to_undirected(data.edge_index)
			data = self.pre_transform(data)

		torch.save(self.collate([data]), self.processed_paths[0])


	def __repr__(self):
		return '{}()'.format(self.__class__.__name__)