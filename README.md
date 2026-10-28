# Community Detection in a Real-World Network

A Python notebook study of the `socfb-Caltech36` Facebook friendship network, using Louvain modularity optimization and Girvan–Newman with approximate edge betweenness. It explores graph structure, centrality, community sizes, modularity, and visualizations.

**Start reading:** [Notebook](Project2_Community_Detection.ipynb) · [Markdown report](Project2_Community_Detection.md) · [Figures](figures/)

The repository contains saved analysis outputs. **The two methods use different graph scopes**, so the reported scores and runtimes are not a controlled, same-graph benchmark. The implementation details below should guide interpretation of the report's method-comparison prose.

## What was analyzed

| Method | Input graph | Procedure |
|---|---|---|
| Louvain | Full graph: 769 nodes, 16,656 undirected edges | `community_louvain.best_partition(G, random_state=42)` from `python-louvain`; modularity is computed on `G` |
| Girvan–Newman | Largest connected component: 762 nodes, 16,651 edges | NetworkX with a fixed-seed approximate edge-betweenness scorer; first partitions with `k = 2, 3, 4`; modularity is computed on `G_lcc` |

The Girvan–Newman scorer samples up to 80 source nodes with seed 42. Its reported time is cumulative from the start of the partition generator, not a separate run for each `k`. Both community visualizations display the largest connected component, even though Louvain is fitted on the full graph.

The saved notebook records 11 Louvain communities with modularity about 0.4009. The tested Girvan–Newman splits have sizes `[761, 1]`, `[760, 1, 1]`, and `[759, 1, 1, 1]`, with modularity near zero. These are observations from this dataset and configuration; they do not establish a general ranking of the algorithms. The edge-only input cannot verify social labels such as residence or class year.

## Dataset and source

- Dataset: [`socfb-Caltech36` on NetworkRepository](https://networkrepository.com/socfb_Caltech36.php).
- Download used by the notebook: [socfb-Caltech36.zip](https://nrvis.com/download/data/socfb/socfb-Caltech36.zip).
- Input: an undirected, unweighted graph loaded from the symmetric MatrixMarket file `socfb-Caltech36.mtx` inside the ZIP.
- The loader creates all 769 nodes, ignores self-loops, and stores each undirected edge once. The source page lists approximately 33.3K edges; this analysis uses the 16,656 unique undirected edges represented by the `.mtx` file.

The notebook reuses `data/socfb-Caltech36.zip` when present and downloads it if absent. The archive includes NetworkRepository's acknowledgement notice and a reference to Ryan A. Rossi and Nesreen K. Ahmed, *The Network Data Repository with Interactive Graph Analytics and Visualization* (AAAI 2015). Preserve that notice and consult the dataset page for additional source requirements.

## Run locally

The saved notebook metadata records **Python 3.12.13**. [`requirements.txt`](requirements.txt) lists minimum package versions rather than a locked environment, so a fresh installation may resolve different versions and produce different timings or partitions.

From a fresh checkout, using Python 3.12 and a virtual environment (macOS/Linux shell):

```bash
git clone https://github.com/kris0516/project2-community-detection.git
cd project2-community-detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook Project2_Community_Detection.ipynb
```

On Windows, activate the same environment with `.venv\Scripts\Activate.ps1` in PowerShell. Start Jupyter in the repository root: the notebook uses `Path.cwd()` to locate `data/` and `figures/`.

Run the cells in order when you want to reproduce the analysis. Package installation requires network access; dataset loading also requires it if the cached ZIP is absent. The setup cell installs missing packages into the active kernel. The analysis saves figures into `figures/` and can replace the existing files. Use the saved notebook or Markdown report for reading without execution. Exact Girvan–Newman on dense graphs can be costly; this notebook uses the approximate scorer and stops at four communities.

## Repository guide

| Path | Purpose |
|---|---|
| [Project2_Community_Detection.ipynb](Project2_Community_Detection.ipynb) | Analysis code, explanations, and saved outputs |
| [Project2_Community_Detection.md](Project2_Community_Detection.md) | Readable fallback when GitHub's notebook preview is unavailable |
| [figures/](figures/) | Ten saved analysis figures |
| [data/socfb-Caltech36.zip](data/socfb-Caltech36.zip) | Existing third-party dataset archive |
| [requirements.txt](requirements.txt) | Package minimum versions |

## Contributions and licensing status

See [CONTRIBUTING.md](CONTRIBUTING.md) for reporting reproducibility issues, correcting explanations, or proposing a comparison on a shared graph. Include the commit, environment, graph scope, and commands used in an [issue](https://github.com/kris0516/project2-community-detection/issues) or pull request.

No repository-level `LICENSE` file is currently included. The bundled dataset has separate source and acknowledgement requirements; its redistribution terms still need to be confirmed before choosing a project license. No license is granted by this README, and the existing archive is retained unchanged.
