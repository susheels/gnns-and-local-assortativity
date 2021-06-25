import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GINConv, SAGEConv

from wrgat import WeightedRGATConv, GATConv
from wrgcn import WeightedRGCNConv


class WRGAT(torch.nn.Module):
	def __init__(self, num_features, num_classes, num_relations, dims=16, drop=0, root=True):
		super(WRGAT, self).__init__()
		self.conv1 = WeightedRGATConv(num_features, dims, num_relations=num_relations, root_weight=root)
		self.conv2 = WeightedRGATConv(dims, num_classes, num_relations=num_relations, root_weight=root)

		self.drop = torch.nn.Dropout(p=drop)
	def forward(self, x, edge_index, edge_weight, edge_color):

		x = F.relu(self.conv1(x, edge_index,edge_weight=edge_weight,edge_type=edge_color))
		x = self.drop(x)

		x = self.conv2(x, edge_index,edge_weight=edge_weight, edge_type=edge_color)

		return F.log_softmax(x, dim=1),x




class WRGCN(torch.nn.Module):
	def __init__(self, num_features, num_classes, num_relations, dims=16, drop=0, root=True):
		super(WRGCN, self).__init__()
		self.conv1 = WeightedRGCNConv(num_features, dims, num_relations=num_relations, num_bases=None, root_weight=root)
		self.conv2 = WeightedRGCNConv(dims, num_classes, num_relations=num_relations, num_bases=None, root_weight=root)

		self.drop = torch.nn.Dropout(p=drop)
	def forward(self, x, edge_index, edge_weight, edge_color):

		x = F.relu(self.conv1(x, edge_index,edge_weight=edge_weight,edge_type=edge_color))
		x = self.drop(x)

		x = self.conv2(x, edge_index,edge_weight=edge_weight, edge_type=edge_color)

		return F.log_softmax(x, dim=1),x


class GAT(torch.nn.Module):
	def __init__(self,num_features, num_classes, dims, drop=0.0):
		super(GAT, self).__init__()
		heads = 8
		self.conv1 = GATConv(num_features,dims, heads=heads, dropout=0.3, concat=False)
		# On the Pubmed dataset, use heads=8 in conv2.
		self.conv2 = GATConv(dims, num_classes, heads=heads, concat=False,
							 dropout=0.3)
		self.drop = torch.nn.Dropout(p=drop)
	def forward(self,x, edge_index):
		x = F.elu(self.conv1(x, edge_index))
		x = self.drop(x)
		x = self.conv2(x, edge_index)
		return F.log_softmax(x, dim=1), x

class GIN(torch.nn.Module):
	def __init__(self, num_features, num_classes, dim=16, drop=0.5):
		super(GIN, self).__init__()

		nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
		self.conv1 = GINConv(nn1)
		self.bn1 = torch.nn.BatchNorm1d(dim)

		nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, num_classes))
		self.conv2 = GINConv(nn2)
		self.bn2 = torch.nn.BatchNorm1d(num_classes)

		self.drop = torch.nn.Dropout(p=drop)

	def forward(self, x, edge_index):
		x = F.relu(self.conv1(x, edge_index))
		x = self.bn1(x)
		x = self.drop(x)
		x = F.relu(self.conv2(x, edge_index))
		x = self.bn2(x)
		return F.log_softmax(x, dim=1), x

class GCN(torch.nn.Module):
	def __init__(self, num_features, num_classes, dim=16, drop=0.5):
		super(GCN, self).__init__()
		self.conv1 = GCNConv(num_features, dim)
		self.conv2 = GCNConv(dim, num_classes)
		self.drop = torch.nn.Dropout(p=drop)
	def forward(self, x, edge_index):
		x = F.relu(self.conv1(x, edge_index))
		x = self.drop(x)
		x = self.conv2(x, edge_index)
		return F.log_softmax(x, dim=1),x

class SAGE(torch.nn.Module):
	def __init__(self, num_features, num_classes, dim=16, drop=0.5):
		super(SAGE, self).__init__()
		self.conv1 = SAGEConv(num_features, dim)
		self.conv2 = SAGEConv(dim, num_classes)
		self.drop = torch.nn.Dropout(p=drop)
	def forward(self, x, edge_index):
		x = F.relu(self.conv1(x, edge_index))
		x = self.drop(x)
		x = self.conv2(x, edge_index)
		return F.log_softmax(x, dim=1),x


class MLP(torch.nn.Module):
	def __init__(self, num_features, num_classes, dims=16):
		super(MLP, self).__init__()
		self.mlp = torch.nn.Sequential(
			torch.nn.Linear(num_features, dims), torch.nn.ReLU(),
			torch.nn.Linear(dims, num_classes))
	def forward(self, x, edge_index):
		x = self.mlp(x)
		return F.log_softmax(x, dim=1),x