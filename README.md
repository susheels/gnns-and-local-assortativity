# GNNs and Local Assortativity  - [Paper](https://arxiv.org/abs/2106.06586) | [Slides]() | [Talk]()

## Introduction
This repo contains a reference implementation for the ACM SIGKDD 2021 paper "Breaking the Limit of Graph Neural Networks by Improving the Assortativity of Graphs with Local Mixing Patterns".
The paper is available on [arxiv](https://arxiv.org/abs/2106.06586) and [ACM DL]().

## Pipeline
<img src="https://raw.githubusercontent.com/susheels/gnns-and-local-assortativity/main/figures/wrgat_pipeline.png"/>

## Requirements and Environment Setup
Code developed and tested in Python 3.8.8 using PyTorch 1.8. Please refer to their official websites for installation and setup. 
GNN models are built using Pytorch Geometric library. Please make sure to install that library first. Information can be found [here](https://github.com/rusty1s/pytorch_geometric).
Some major requirements are given below.
```
numpy==1.20.1
torch==1.8.1
tqdm==4.60.0
networkx==2.5.1
six==1.15.0
fastdtw==0.3.4
scipy==1.6.2
scikit-learn==0.24.1
torch-geometric==1.7.0
```
    
## Datasets

The package `datasets` contains the modules required for loading the datasets used for the experiments and analysis.

WebKB and Wikipedia datasets are from [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/master/splits) and Citation network is loaded directly from Pytorch Geometric library. These datasets are automatically downloaded. 

Air-Traffic dataset is from struc2vec [2] and Internet BGP dataset is from Hou et. al [3]. 
They are made available in the `raw_data_src` folder. Please unzip them by following the commands from the project root :
```
cd raw_data_src
unzip airports_dataset_dump.zip
unzip bgp_data_dump.zip
```

Please refer to the appendix of our [paper](https://arxiv.org/abs/2106.06586) for more details regarding dataset summary and statistics.

## Local Assortativity Plots
File `local_assortativity_plots.ipynb` shows how to plot local assortativity for our datasets. 

More generally the functions in `gnnutils.py` for finding global and local assortativity can be used for any dataset of choice as long as they are loaded in NetworkX or Pytorch Geometric format (using the `to_networkx()` function).

<img src="https://raw.githubusercontent.com/susheels/gnns-and-local-assortativity/main/figures/local_assortativity_example.png"/>



## Training WRGAT and WRGCN

To train and evaluate our method which uses structure and proximity information, use the `exp.py` file. For example running `python exp.py --help` provides :

```
usage: exp.py [-h] --dataset DATASET --model MODEL [--original_edges] [--original_edges_weight ORIGINAL_EDGES_WEIGHT] [--filter_structure_relation] [--filter_structure_relation_number FILTER_STRUCTURE_RELATION_NUMBER] [--run_times RUN_TIMES] [--dims DIMS]
              [--epochs EPOCHS] [--drop DROP] [--custom_masks] [--st_thres ST_THRES] [--lr LR]

WRGAT/WRGCN (structure + proximity) Experiments

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset
  --model MODEL         GNN Model
  --original_edges
  --original_edges_weight ORIGINAL_EDGES_WEIGHT
  --filter_structure_relation
  --filter_structure_relation_number FILTER_STRUCTURE_RELATION_NUMBER
  --run_times RUN_TIMES
  --dims DIMS           hidden dims
  --epochs EPOCHS
  --drop DROP           dropout
  --custom_masks        custom train/val/test masks
  --st_thres ST_THRES   edge weight threshold
  --lr LR               learning rate
``` 

Hyperparameter settings are given in the appendix of the paper.

<img src="https://raw.githubusercontent.com/susheels/gnns-and-local-assortativity/main/figures/gnn_local_assortativity.png"/>

## Acknowledgements

The structural similarity measure is based on struc2vec [2] and notion of local assortativity is from Peel et. al [4].

##
Please cite our paper if you use this code in your own work.
```
@article{suresh2021breaking,
  title={Breaking the Limit of Graph Neural Networks by Improving the Assortativity of Graphs with Local Mixing Patterns},
  author={Suresh, Susheel and Budde, Vinith and Neville, Jennifer and Li, Pan and Ma, Jianzhu},
  journal={arXiv preprint arXiv:2106.06586},
  year={2021}
}
```

## References
	[1] Paszke, Adam, et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." Advances in Neural Information Processing Systems 32 (2019): 8026-8037.
    [2] Ribeiro, Leonardo FR, Pedro HP Saverese, and Daniel R. Figueiredo. "struc2vec: Learning node representations from structural identity." Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining. 2017.
    [3] Yifan Hou, Jian Zhang, James Cheng, Kaili Ma, Richard TB Ma, Hongzhi Chen, and Ming-Chang Yang. 2019. Measuring and improving the use of graph information in graph neural networks. In ICLR
    [4] Peel, Leto, Jean-Charles Delvenne, and Renaud Lambiotte. "Multiscale mixing patterns in networks." Proceedings of the National Academy of Sciences 115.16 (2018): 4057-4062.