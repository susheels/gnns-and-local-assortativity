# GNNs and Local Assortativity  - [Paper](https://arxiv.org/abs/2106.06586) | [Slides]() | [Talk]()

## Introduction
This repo contains a reference implementation for the ACM SIGKDD 2021 paper "Breaking the Limit of Graph Neural Networks by Improving the Assortativity of Graphs with Local Mixing Patterns".
The paper is available on [arxiv](https://arxiv.org/abs/2106.06586) and [ACM DL]().



## Requirements and Environment Setup
Code developed and tested in Python 3.8.8 using PyTorch 1.8. Please refer to their official websites for installation and setup. 

Some major requirements are given below.
```
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

<img src=""/>

## Training WRGAT and WRGCN


## Acknowledgements

The structural similarity measure is based on struc2vec [2] and notion of local assortativity is from Peel et. al [4].

##
Please cite our paper if you use this code in your own work.
```
@article{suresh2021adversarial,
  title={Adversarial Graph Augmentation to Improve Graph Contrastive Learning},
  author={Suresh, Susheel and Li, Pan and Hao, Cong and Neville, Jennifer},
  journal={arXiv preprint arXiv:2106.05819},
  year={2021}
}
```

## References
	[1] Paszke, Adam, et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." Advances in Neural Information Processing Systems 32 (2019): 8026-8037.
    [2] Ribeiro, Leonardo FR, Pedro HP Saverese, and Daniel R. Figueiredo. "struc2vec: Learning node representations from structural identity." Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining. 2017.
    [3] Yifan Hou, Jian Zhang, James Cheng, Kaili Ma, Richard TB Ma, Hongzhi Chen, and Ming-Chang Yang. 2019. Measuring and improving the use of graph information in graph neural networks. In ICLR
    [4] Peel, Leto, Jean-Charles Delvenne, and Renaud Lambiotte. "Multiscale mixing patterns in networks." Proceedings of the National Academy of Sciences 115.16 (2018): 4057-4062.