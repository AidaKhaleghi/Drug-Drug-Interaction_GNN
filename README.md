# Drug-Drug Interaction Prediction with a GNN

Predicts drug-drug interactions as a link prediction problem on the [`ogbl-ddi`](https://ogb.stanford.edu/docs/linkprop/#ogbl-ddi) graph. Each drug is a node with a learned embedding, refined by two GraphSAGE layers. A small MLP scores a drug pair from the element-wise product of its two embeddings. The metric is Hits@20.

## Project structure

```
src/ddi_gnn/
  data.py         Loads ogbl-ddi, builds the graph and the train/valid/test pairs
  model.py        GraphSAGE encoder and link predictor
  losses.py       Binary log loss and AUC loss
  metrics.py      Hits@K and the metric logger
  train.py        Training loop, best-epoch checkpointing, saving and loading runs
  experiments.py  Multi-seed runs, sweeps and result summaries
  plotting.py     Learning curves, sweep plots and the graph plot
notebooks/
  01_data_exploration.ipynb
  02_experiments.ipynb   Loss, embedding size and hyperparameter experiments
results/                 Tables and figures written by the experiments notebook
```

## Installation

```bash
pip install -e .
```

The notebooks run on Google Colab (GPU recommended).

## Evaluation protocol

- All choices (loss, embedding size, dropout, ...) are made with the **validation** split only.
- Each run keeps the weights of its **best validation epoch**.
- Every setting is trained with several seeds and reported as **mean ± std**.
- The **test** split is scored only for the final selected setting (`evaluate_test=True`).
- Train Hits@20 is measured in eval mode (no dropout), so it can be compared with validation.

## Results

Produced by `notebooks/02_experiments.ipynb` (Hits@20 at the best validation epoch, mean ± std over seeds). The full tables and figures are in [`results/`](results/).

<!-- Paste the content of results/summary.md here after running the notebook. -->

## Usage

```python
import torch

from ddi_gnn.data import load_ddi
from ddi_gnn.experiments import run_seeds

data = load_ddi(root="datasets", node_limit=500, generator=torch.Generator().manual_seed(42))
model_params = {
    "embedding_dim": 256,
    "dropout_rate": 0.3,
    "last_layer_self": False,
    "degree_norm": False,
}

# Model selection: validation only.
runs = run_seeds("my_model", data, model_params, seeds=[0, 1, 2], loss="auc", save_dir="models")

# Final report: also score the test split.
runs = run_seeds("my_model_final", data, model_params, seeds=[0, 1, 2], loss="auc",
                 save_dir="models", evaluate_test=True)
```
