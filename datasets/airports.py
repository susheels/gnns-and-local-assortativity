import shutil

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import *


def get_degrees(G):
	num_nodes = G.number_of_nodes()
	return np.array([G.degree[i] for i in range(num_nodes)])


class Airports(InMemoryDataset):
	def __init__(self, root, dataset_name, transform=None, pre_transform=None):
		self.dataset_name  = dataset_name
		self.dump_location = "raw_data_src/airports_dataset_dump"
		super(Airports, self).__init__(root,transform, pre_transform)
		self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def raw_file_names(self):
		return [self.dataset_name+'-airports.edgelist', 'labels-'+self.dataset_name+'-airports.txt']

	@property
	def processed_file_names(self):
		return 'data.pt'

	def download(self):
		for name in self.raw_file_names:
			source = self.dump_location + '/' + name
			shutil.copy(source, self.raw_dir)

	def process(self):

		fin_labels = open(self.raw_paths[1])
		labels = []
		node_id_mapping = dict()
		node_id_labels_dict = dict()
		for new_id, line in enumerate(fin_labels.readlines()[1:]): # first line is header so ignore
			old_id, label = line.strip().split()
			labels.append(int(label))
			node_id_mapping[old_id] = new_id
			node_id_labels_dict[new_id] = int(label)
		fin_labels.close()

		edges = []
		fin_edges = open(self.raw_paths[0])
		for line in fin_edges.readlines():
			node1, node2 = line.strip().split()[:2]
			edges.append([node_id_mapping[node1], node_id_mapping[node2]])
		fin_edges.close()

		networkx_graph = nx.Graph(edges)

		print("No. of Nodes: ",networkx_graph.number_of_nodes())
		print("No. of edges: ",networkx_graph.number_of_edges())


		attr = {}
		for node in networkx_graph.nodes():
			deg = networkx_graph.degree(node)
			attr[node] = {"y": node_id_labels_dict[node], "x": [float(deg)]}
		nx.set_node_attributes(networkx_graph, attr)
		data = from_networkx(networkx_graph)


		if self.pre_filter is not None:
			data = self.pre_filter(data)

		if self.pre_transform is not None:
			data = self.pre_transform(data)

		torch.save(self.collate([data]), self.processed_paths[0])


	def __repr__(self):
		return '{}()'.format(self.__class__.__name__)