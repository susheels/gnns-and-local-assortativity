import copy

import networkx as nx
import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn.functional as F
from networkx.utils import dict_to_numpy_array
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_remaining_self_loops
from tqdm import tqdm

from build_multigraph import build_pyg_struc_multigraph
from datasets import WebKB, WikipediaNetwork, FilmNetwork, BGP, Airports


def load_airports(dataset):
	assert dataset in ["brazil", "europe", "usa"]
	og = Airports("original_datasets/airports_dataset/"+dataset, dataset_name=dataset)[0]
	st = Airports(root="datasets_py_geom_format/airports_dataset/"+dataset, dataset_name=dataset,
                   pre_transform=build_pyg_struc_multigraph)[0]
	return og, st


def load_bgp(dataset):
	assert dataset in ["bgp"]
	og = BGP(root="original_datasets/bgp_dataset")[0]
	st = BGP(root="datasets_py_geom_format/bgp_dataset", pre_transform=build_pyg_struc_multigraph)[0]
	return og, st

def load_film(dataset):
	assert dataset in ["film"]
	og = FilmNetwork(root="original_datasets/film", name=dataset)[0]
	st = FilmNetwork(root="datasets_py_geom_format/film", name=dataset,
	                      pre_transform=build_pyg_struc_multigraph)[0]
	return og, st

def load_wiki(dataset):
	assert dataset in ["chameleon", "squirrel"]
	og = WikipediaNetwork(root="original_datasets/wiki", name=dataset)[0]
	st = WikipediaNetwork(root="datasets_py_geom_format/wiki", name=dataset,
	           pre_transform=build_pyg_struc_multigraph)[0]

	return og, st

def load_webkb(dataset):
	assert dataset in ["cornell", "texas", "wisconsin"]
	og = WebKB(root="original_datasets/webkb", name=dataset)[0]
	st = WebKB(root="datasets_py_geom_format/webkb", name=dataset,
	           pre_transform=build_pyg_struc_multigraph)[0]

	return og, st

def load_planetoid(dataset):
	assert dataset in ["cora", "citeseer", "pubmed"]
	og = Planetoid(root="original_datasets/planetoid", name=dataset, split="public")[0]
	st = Planetoid(root="datasets_py_geom_format/planetoid", name=dataset, split="public",
	               pre_transform=build_pyg_struc_multigraph)[0]

	return og, st


def structure_edge_weight_threshold(data, threshold):
	data = copy.deepcopy(data)
	mask = data.edge_weight > threshold
	data.edge_weight = data.edge_weight[mask]
	data.edge_index = data.edge_index[:, mask]
	data.edge_color = data.edge_color[mask]
	return data


def add_original_graph(og_data, st_data, weight=1.0):
	st_data = copy.deepcopy(st_data)
	e_i = torch.cat((og_data.edge_index,st_data.edge_index), dim=1)
	st_data.edge_color = st_data.edge_color + 1
	e_c = torch.cat((torch.zeros(og_data.edge_index.shape[1], dtype=torch.long),st_data.edge_color), dim=0)
	e_w = torch.cat((torch.ones(og_data.edge_index.shape[1], dtype=torch.float)*weight,st_data.edge_weight), dim=0)
	st_data.edge_index = e_i
	st_data.edge_color = e_c
	st_data.edge_weight = e_w
	return st_data

def train(model, train_data, mask, optimizer, device, use_structure_info):
	model.train()
	optimizer.zero_grad()
	mask = mask.to(device)
	true_label = train_data.y.to(device)
	if use_structure_info:
		out, _ = model(train_data.x.to(device),train_data.edge_index.to(device),
				train_data.edge_weight.to(device),train_data.edge_color.to(device))
	else:
		out, _ = model(train_data.x.to(device),train_data.edge_index.to(device))
	loss = F.nll_loss(out[mask],true_label[mask])
	pred = out.max(1)[1].cpu()[mask]
	acc = pred.eq(train_data.y[mask]).sum().item() / len(train_data.y[mask])
	loss.backward()
	optimizer.step()
	return loss.item(), acc



@torch.no_grad()
def test(model, test_data, mask, device, use_structure_info):
	model.eval()
	mask = mask.to(device)
	if use_structure_info:
		logits, raw = model(test_data.x.to(device), test_data.edge_index.to(device),
				test_data.edge_weight.to(device), test_data.edge_color.to(device))
	else:
		logits, raw,  = model(test_data.x.to(device), test_data.edge_index.to(device))
	pred = logits.max(1)[1].cpu()[mask]
	acc = pred.eq(test_data.y[mask]).sum().item() / len(test_data.y[mask])
	pred = pred.numpy()
	y_true = test_data.y[mask].numpy()
	f1 = f1_score(y_true, pred, average='micro')
	return acc, f1

def filter_relations(data, num_relations, rel_last):
	if rel_last:
		l = data.edge_color.unique(sorted=True).tolist()
		mask_l = l[-num_relations:]
		mask = data.edge_color == mask_l[0]
		for c in mask_l[1:]:
			mask = mask | (data.edge_color == c)
	else:
		mask = data.edge_color < (num_relations + 1)

	data.edge_index = data.edge_index[:, mask]
	data.edge_weight = data.edge_weight[mask]
	data.edge_color = data.edge_color[mask]
	return data


def make_masks(data, val_test_ratio=0.2, stratify=False):
	data = copy.deepcopy(data)
	n_nodes = data.x.shape[0]
	all_nodes_idx = np.arange(n_nodes)
	all_y = data.y.numpy()
	if stratify:
		train, test_idx, y_train, _ = train_test_split(
			all_nodes_idx, all_y, test_size=0.2, stratify=all_y)

		train_idx, val_idx, _, _ = train_test_split(
			train, y_train, test_size=0.25, stratify=y_train)

	else:
		val_test_num = 2 * int(val_test_ratio * data.x.shape[0])
		val_test_idx = np.random.choice(n_nodes, (val_test_num,), replace=False)
		val_idx = val_test_idx[:int(val_test_num / 2)]
		test_idx = val_test_idx[int(val_test_num / 2):]

	val_mask = np.zeros(n_nodes)
	val_mask[val_idx] = 1
	test_mask = np.zeros(n_nodes)
	test_mask[test_idx] = 1
	val_mask = torch.tensor(val_mask, dtype=torch.bool)
	test_mask = torch.tensor(test_mask, dtype=torch.bool)
	val_test_mask = val_mask | test_mask
	train_mask = ~val_test_mask
	data.train_mask = train_mask
	data.val_mask = val_mask
	data.test_mask = test_mask
	return data

def create_self_loops(data):
	orig_relations = len(data.edge_color.unique())
	data.edge_index, data.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_weight,
																 fill_value=1.0)
	row, col = data.edge_index[0], data.edge_index[1]
	mask = row == col
	tmp = torch.full(mask.nonzero().shape, orig_relations + 1, dtype=torch.long).squeeze()
	data.edge_color = torch.cat([data.edge_color, tmp], dim=0)
	return data


# create adjacency matrix and degree sequence
def createA(E, n, m, undir=True):
	if undir:
		G = nx.Graph()
		G.add_nodes_from(range(n))
		G.add_edges_from(list(E))
		A = nx.to_scipy_sparse_matrix(G)
	else:
		A = sparse.coo_matrix((np.ones(m), (E[:, 0], E[:, 1])),
							  shape=(n, n)).tocsc()
	degree = np.array(A.sum(1)).flatten()

	return A, degree

# calculate the stationary distributions of a random walk with restart
# for different probabilties of restart (using the pagerank as a function
# approach)
def calculateRWRrange(A, degree, i, prs, n, trans=True, maxIter=1000):
	pr = prs[-1]
	D = sparse.diags(1./degree, 0, format='csc')
	W = D*A
	diff = 1
	it = 1

	F = np.zeros(n)
	Fall = np.zeros((n, len(prs)))
	F[i] = 1
	Fall[i, :] = 1
	Fold = F.copy()
	T = F.copy()

	if trans:
		W = W.T

	oneminuspr = 1-pr

	while diff > 1e-9:
		F = pr*W.dot(F)
		F[i] += oneminuspr
		Fall += np.outer((F-Fold), (prs/pr)**it)
		T += (F-Fold)/((it+1)*(pr**it))

		diff = np.sum((F-Fold)**2)
		it += 1
		if it > maxIter:
			print(i, "max iterations exceeded")
			diff = 0
		Fold = F.copy()

	return Fall, T, it


def localAssortF(G, M, pr=np.arange(0., 1., 0.1), undir=True, missingValue=-1, edge_attribute=None):
	n = len(M)
	ncomp = (M != missingValue).sum()
	# m = len(E)
	m = G.number_of_edges()

	# A, degree = createA(E, n, m, undir)
	if edge_attribute is None:
		A = nx.to_scipy_sparse_matrix(G, weight=None)
	else:
		A = nx.to_scipy_sparse_matrix(G, weight=edge_attribute)

	degree = np.array(A.sum(1)).flatten()

	D = sparse.diags(1. / degree, 0, format='csc')
	W = D.dot(A)
	c = len(np.unique(M))
	if ncomp < n:
		c -= 1

	# calculate node weights for how "complete" the
	# metadata is around the node
	Z = np.zeros(n)
	Z[M == missingValue] = 1.
	Z = W.dot(Z) / degree

	values = np.ones(ncomp)
	yi = (M != missingValue).nonzero()[0]
	yj = M[M != missingValue]
	Y = sparse.coo_matrix((values, (yi, yj)), shape=(n, c)).tocsc()

	assortM = np.empty((n, len(pr)))
	assortT = np.empty(n)

	eij_glob = np.array(Y.T.dot(A.dot(Y)).todense())
	eij_glob /= np.sum(eij_glob)
	ab_glob = np.sum(eij_glob.sum(1) * eij_glob.sum(0))

	WY = W.dot(Y).tocsc()

	print("start iteration")

	for i in tqdm(range(n)):
		pis, ti, it = calculateRWRrange(A, degree, i, pr, n)
		# for ii, pri in enumerate(pr):
		# 	pi = pis[:, ii]
		#
		# 	YPI = sparse.coo_matrix((pi[M != missingValue],
		# 							 (M[M != missingValue],
		# 							  np.arange(n)[M != missingValue])),
		# 							shape=(c, n)).tocsr()
		#
		# 	trace_e = np.trace(YPI.dot(WY).toarray())
		# 	assortM[i, ii] = trace_e
		YPI = sparse.coo_matrix((ti[M != missingValue], (M[M != missingValue],
														 np.arange(n)[M != missingValue])),
								shape=(c, n)).tocsr()
		e_gh = YPI.dot(WY).toarray()
		Z[i] = np.sum(e_gh)
		e_gh /= np.sum(e_gh)
		trace_e = np.trace(e_gh)
		assortT[i] = trace_e

	# assortM -= ab_glob
	# assortM /= (1. - ab_glob + 1e-200)

	assortT -= ab_glob
	assortT /= (1. - ab_glob + 1e-200)

	return assortM, assortT, Z




def mixing_dict(xy, normalized=True):
	d = {}
	psum = 0.0
	for x, y, w in xy:
		if x not in d:
			d[x] = {}
		if y not in d:
			d[y] = {}
		v = d[x].get(y, 0)
		d[x][y] = v + w
		psum += w

	if normalized:
		for k, jdict in d.items():
			for j in jdict:
				jdict[j] /= psum
	return d

def node_attribute_xy(G, attribute, edge_attribute=None, nodes=None):
	if nodes is None:
		nodes = set(G)
	else:
		nodes = set(nodes)
	Gnodes = G.nodes
	for u, nbrsdict in G.adjacency():
		if u not in nodes:
			continue
		uattr = Gnodes[u].get(attribute, None)
		if G.is_multigraph():
			raise NotImplementedError
		else:
			for v, eattr in nbrsdict.items():
				vattr = Gnodes[v].get(attribute, None)
				if edge_attribute is None:
					yield (uattr, vattr, 1)
				else:
					edge_data = G.get_edge_data(u, v)
					yield (uattr, vattr, edge_data[edge_attribute])


def global_assortativity(networkx_graph, labels, weights=None):
	attr_dict = {}
	for i in networkx_graph.nodes():
		attr_dict[i] = labels[i]

	nx.set_node_attributes(networkx_graph, attr_dict, "label")
	if weights is None:
		xy_iter = node_attribute_xy(networkx_graph, "label", edge_attribute=None)
		d = mixing_dict(xy_iter)
		M = dict_to_numpy_array(d, mapping=None)
		s = (M @ M).sum()
		t = M.trace()
		r = (t - s) / (1 - s)
	else:
		edge_attr = {}
		for i, e in enumerate(networkx_graph.edges()):
			edge_attr[e] = weights[i]
		nx.set_edge_attributes(networkx_graph, edge_attr, "weight")

		xy_iter = node_attribute_xy(networkx_graph, "label", edge_attribute="weight")
		d = mixing_dict(xy_iter)
		M = dict_to_numpy_array(d, mapping=None)
		s = (M @ M).sum()
		t = M.trace()
		r = (t - s) / (1 - s)

	return r, M

def local_assortativity(networkx_graph, labels, weights=None):
	if weights is None:
		assort_m, assort_t, z = localAssortF(networkx_graph, np.array(labels))
	else:
		edge_attr = {}
		for i, e in enumerate(networkx_graph.edges()):
			edge_attr[e] = weights[i]
		nx.set_edge_attributes(networkx_graph, edge_attr, "weight")
		assort_m, assort_t, z = localAssortF(networkx_graph, np.array(labels),edge_attribute="weight")

	return assort_m, assort_t, z
