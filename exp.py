import argparse
import copy
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gnnutils import make_masks, train, test, add_original_graph, load_webkb, load_planetoid, load_wiki, load_bgp, \
	load_film, structure_edge_weight_threshold
from models import WRGAT, WRGCN

MODEl_DICT = {"WRGAT": WRGAT, "WRGCN": WRGCN}

def filter_rels(data, r):
	data = copy.deepcopy(data)
	mask = data.edge_color <= r
	data.edge_index = data.edge_index[:, mask]
	data.edge_weight = data.edge_weight[mask]
	data.edge_color = data.edge_color[mask]
	return data

def run(i, data, num_features, num_classes, num_relations, isbgp):
	if args.model in MODEl_DICT:
		model = MODEl_DICT[args.model](num_features, num_classes, num_relations, dims=args.dims, drop=args.drop).to(device)
	else:
		print("Model not supported")
		raise NotImplementedError

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)

	if args.custom_masks:
		# create random mask
		data = make_masks(data, val_test_ratio=0.2)
		train_mask = data.train_mask
		val_mask = data.val_mask
		test_mask = data.test_mask
	else:
		if isbgp:
			train_mask = data.train_mask
			val_mask = data.val_mask
			test_mask = data.test_mask
		else:
			# use existing masks, run number i is used to index.
			# 10 runs are supported as geom-gcn has only 10 splits.
			assert i < 10
			train_mask = data.train_mask[:, i]
			val_mask = data.val_mask[:, i]
			test_mask = data.test_mask[:,i]

	best_val_acc = 0
	best_model = None
	pat = 20
	best_epoch = 0
	for epoch in range(1, args.epochs+1):

		train_loss, train_acc = train(model, data, train_mask, optimizer, device, use_structure_info=True)
		if epoch % 10 == 0:
			## valid
			valid_acc, valid_f1 = test(model, data, val_mask, device, use_structure_info=True)

			if valid_acc > best_val_acc:
				best_val_acc = valid_acc
				# print("val metric improving")
				best_model = model
				best_epoch = epoch
				pat = (pat + 1) if (pat < 5) else pat
			else:
				pat -= 1

			logging.info(
				'Epoch: {:03d}, Train Loss: {:.4f}, Train Acc: {:0.4f}, Val Acc: {:0.4f} '.format(
					epoch, train_loss, train_acc, valid_acc))

			if pat < 0:
				logging.info("validation patience reached ... finish training")
				break


	## Testing
	test_acc, test_f1 = test(best_model, data, test_mask, device, use_structure_info=True)
	logging.info('Best Val Epoch: {:03d}, Best Val Acc: {:0.4f}, Test Acc: {:0.4f} '.format(
		best_epoch, best_val_acc, test_acc))


	return test_acc, test_f1

def main():

	timestr = time.strftime("%Y%m%d-%H%M%S")
	log_file = args.dataset +"-" +timestr + ".log"
	Path("./exp_logs").mkdir(parents=True, exist_ok=True)
	logging.basicConfig(filename="exp_logs/"+log_file, filemode="w", level=logging.INFO)
	logging.info("Starting on device: %s", device)
	logging.info("Config: %s ", args)
	isbgp = False
	if args.dataset in ['cornell', 'texas', 'wisconsin']:
		assert args.custom_masks == False
		og_data, st_data = load_webkb(args.dataset)
	elif args.dataset in ['cora', 'citeseer', 'pubmed']:
		assert args.custom_masks == True
		og_data, st_data = load_planetoid(args.dataset)
	elif args.dataset in ["chameleon", "squirrel"]:
		assert args.custom_masks == False
		og_data, st_data = load_wiki(args.dataset)
	elif args.dataset in ["bgp"]:
		# for bgp only one split given from original paper is used.
		assert args.custom_masks == False
		isbgp = True
		og_data, st_data = load_bgp(args.dataset)
	elif args.dataset in ["film"]:
		assert args.custom_masks == False
		og_data, st_data = load_film(args.dataset)
	else:
		raise NotImplementedError

	if args.filter_structure_relation:
		print("Filtering structure relations with number %d" % args.filter_structure_relation_number)
		st_data = filter_rels(st_data, args.filter_structure_relation_number)

	st_data = structure_edge_weight_threshold(st_data, args.st_thres)

	if args.original_edges:
		print("Adding original graph edges with weight %f" % args.original_edges_weight)
		data = add_original_graph(og_data, st_data, weight=args.original_edges_weight)
	else:
		data = st_data

	num_classes = len(data.y.unique())
	num_features = data.x.shape[1]
	num_relations = len(data.edge_color.unique())

	runs_acc = []
	for i in tqdm(range(args.run_times)):
		acc = run(i, data, num_features, num_classes, num_relations, isbgp)
		runs_acc.append(acc)
	runs_acc = np.array(runs_acc) * 100

	final_msg = "Mean %0.4f, Std %0.4f" % (runs_acc.mean(), runs_acc.std())
	print(final_msg)
	logging.info("%s", runs_acc)
	logging.info("%s", final_msg)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="WRGAT/WRGCN (structure + proximity) Experiments")

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print("Using", device)

	parser.add_argument("--dataset", required=True, help="Dataset")
	parser.add_argument("--model", default="WRGAT", required=True, help="GNN Model")
	parser.add_argument("--original_edges", default=False, action='store_true')
	parser.add_argument("--original_edges_weight", type=float, default=1.0)
	parser.add_argument("--filter_structure_relation", default=False, action='store_true')
	parser.add_argument("--filter_structure_relation_number", type=int, default=10)
	parser.add_argument("--run_times", type=int, default=10)
	parser.add_argument("--dims", type=int, default=128, help="hidden dims")
	parser.add_argument("--epochs", type=int, default=300)
	parser.add_argument("--drop", type=float, default=0.5, help="dropout")
	parser.add_argument("--custom_masks", default=False, action='store_true',help="custom train/val/test masks")
	parser.add_argument("--st_thres", type=float,  default=-math.inf, help="edge weight threshold")
	parser.add_argument("--lr", type=float,  default=1e-2, help="learning rate")

	args = parser.parse_args()
	print(args)
	main()

