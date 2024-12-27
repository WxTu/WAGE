## Paper

Source code for the paper "WAGE: Weight-Sharing Attribute-Missing Graph Autoencoder" (under review)<br>

Wenxuan Tu, Sihang Zhou, Xinwang Liu, Zhiping Cai, Yawei Zhao, Yue Liu, and Kunlun He<br>

## Installation

Clone this repo.
```bash
git clone https://github.com/WxTu/WAGE.git
```

* Python 3.9.19
* Pytorch 2.0.1
* Numpy 1.26.5
* Sklearn 1.3.0
* Torchvision 0.15.2
* Matplotlib 3.9.9
* DGL 2.0.0 
* Networkx 2.8.8
* PYG 2.5.2

## Preparation

We adopt six datasets in total, including Cora, Citeseer, Pubmed, Amazon Computer, Amazon Photo, and Coauthor-CS. To train WAGE, the dataset should first be selected by configuring the "--dataset" parameter. Subsequently, the code can be executed using `main.py`, with the selected datasets being automatically downloaded from the DGL library.

## Code Structure & Usage

The repository is organized as follows:

- `configs/wage/configs.yml`: defines some hyper-parameters.
- `datasets/pyg_data_utils.py`: downloads and processes the dataset before passing it to the network.
- `models/wage.py`: defines the architecture of the whole network.
- `utils/utils.py`: defines some functions about data processing, evaluation metrics, loss, and others.
- `trains/train_wage.py`: the entry point for training and testing.
- `__init__.py`: initialization files of `wage.py` and `train_wage.py`.
- `main.py`: the entry point for training and testing.

Finally, `main.py` puts all of the above together and may be used to execute an entire training run on these datasets.

<span id="jump2"></span>


## Contact
[wenxuantu@163.com](wenxuantu@163.com)

Any discussions or concerns are welcomed!

## Citation & License
All rights reserved.
Licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0). 

The code is released for academic research use only. For commercial use, please contact [wenxuantu@163.com].

## Acknowledgement

X. Chen, S. Chen, J. Yao, et al. Learning on Attribute-Missing Graphs. IEEE TPAMI, 2022.<br/> 
--[https://github.com/xuChenSJTU/SAT-master-online](https://github.com/xuChenSJTU/SAT-master-online)
