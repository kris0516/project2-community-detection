# Project 2: Community Detection in a Real-World Network

This project analyzes the `socfb-Caltech36` Facebook friendship network from NetworkRepository using two community detection methods:

- Louvain method, based on modularity optimization
- Girvan-Newman algorithm, based on edge betweenness

The final notebook is:

[Project2_Community_Detection.ipynb](Project2_Community_Detection.ipynb)

GitHub notebook link:

https://github.com/kris0516/project2-community-detection/blob/main/Project2_Community_Detection.ipynb

## Dataset

- Dataset: `socfb-Caltech36`
- Source: https://networkrepository.com/socfb_Caltech36.php
- Direct download used by the notebook: https://nrvis.com/download/data/socfb/socfb-Caltech36.zip
- Network type: undirected, unweighted Facebook friendship network
- Nodes: 769
- Unique undirected edges in the MatrixMarket file: 16,656

NetworkRepository lists this graph with approximately 33.3K edges because the symmetric relationship may be counted in both directions. The analysis uses the unique undirected edges represented by the `.mtx` file.

## How to Run

From this folder:

```bash
python3 -m pip install -r requirements.txt
jupyter notebook Project2_Community_Detection.ipynb
```

Run all notebook cells from top to bottom. Generated figures are saved in `figures/`, and the downloaded dataset is cached in `data/`.

## Notes

The Girvan-Newman method is much more computationally expensive than Louvain on dense graphs. To keep the notebook reproducible and runnable for submission, the notebook uses NetworkX's Girvan-Newman implementation with a fixed-seed approximate edge-betweenness scorer and evaluates `k = 2, 3, 4` communities on the largest connected component.
