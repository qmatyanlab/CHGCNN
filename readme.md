# Crystal Hypergraph Convolutional Neural Networks

Implementation of [Crystal Hypergraph Convolutional Neural Networks]().

## How to cite

If you use our code or method in your work, please cite it as follows:
```

```

##  Prerequisites

- [pymatgen](https://github.com/materialsproject/pymatgen) - Python Materials Genomics (pymatgen) is a Python library for materials analysis.
- [PyTorch](http://pytorch.org) - An open source deep learning platform.
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) - PyTorch extension library for graph neural networks.

optional:
- [wandb](https://github.com/wandb/wandb) - A visualization tool for experiment tracking for machine learning.

### Training

Run training with the following command:
```bash
python main_nosym.py processed_data_dir
```

### Project Structure

The directory structure of new project looks like this:

```
├── .github                   <- Github Actions workflows
│
├── data                 
│   ├── hypergraph                <- Hypergraph definitions
│   │     ├── hyperedges          <- Defines hyperedge types
│   │     │    ├── bonds.py
│   │     │    ├── hyperedge.py
│   │     │    ├── motifs.py
│   │     │    ├── triplets.py
│   │     │    └── unit_cell.py
│   │     ├── rbf                  <- Radial basis functions
│   │     │    └──  gaussian.py
│   │     ├── hypergraph.py        <- Crystal hypergraph class
│   │     └── neighbor_list.py     <- Generate neighbor lists
│   ├── utilities                 
│   │     ├── collate.py               <- Fix collate in torch-geometric for hyperedges
│   │     ├── fingerprint.py           <- Fix collate in pymatgen for motifs
│   │     └── readme.md
│   └── genereate_nosym.py                <- Process cifs
│
├── model                         
│   ├── convolutions                <- Hypergraph definitions
│   │     ├── inter_conv.py          <- Defines interorder convolution
│   │     └── agg_conv.py            <- Defines neighborhood-aggregation convolution
│   └── chgcnn.py                   <- Model definition
│   
├── main_nosym.py                   <- Runs training loop
└── readme.md
```
