import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from ddi_gnn.data import DdiData, Graph, Pairs
from ddi_gnn.losses import LOSSES
from ddi_gnn.metrics import MetricsLogger, hits_at_k
from ddi_gnn.model import DdiModel
from ddi_gnn.utils import get_device, set_seed


def train_step(model, optimizer, graph: Graph, pairs: Pairs, loss_fn) -> dict:
    """Runs one optimizer step on a batch of pairs."""
    model.train()
    optimizer.zero_grad()

    scores = model(graph, pairs)
    loss = loss_fn(scores)
    loss.backward()
    optimizer.step()

    return {"loss": loss.item(), "hits@20": hits_at_k(scores)}


@torch.no_grad()
def evaluate(model, graph: Graph, pairs: Pairs, loss_fn) -> dict:
    """Scores every pair of a split at once and returns its loss and Hits@20."""
    model.eval()
    scores = model(graph, pairs)
    return {"loss": loss_fn(scores).item(), "hits@20": hits_at_k(scores)}


def train(
    model: DdiModel,
    optimizer: torch.optim.Optimizer,
    data: DdiData,
    loss_fn,
    num_epochs: int,
    batch_size: int,
    generator: torch.Generator,
    eval_every: int = 1,
) -> pd.DataFrame:
    """Trains the model and returns the metric history."""
    logger = MetricsLogger()
    progress = tqdm(range(num_epochs))

    for epoch in progress:
        progress.set_description(f"Epoch {epoch + 1}")

        for batch in data.splits["train"].train_batches(batch_size, generator):
            logger.log(
                "train", train_step(model, optimizer, data.graph, batch, loss_fn)
            )

        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            logger.log(
                "valid", evaluate(model, data.graph, data.splits["valid"], loss_fn)
            )

        logger.flush(epoch)
        progress.set_postfix_str(logger.latest("hits@20"))

    return logger.to_frame()


def run_experiment(
    name: str,
    data: DdiData,
    model_params: dict,
    loss: str = "auc",
    lr: float = 1e-3,
    num_epochs: int = 500,
    batch_size: int = 2048,
    eval_every: int = 25,
    seed: int = 42,
    save_dir: str = "models",
    device: torch.device | None = None,
) -> tuple[DdiModel, pd.DataFrame]:
    """Trains one model from scratch, evaluates it on the test split and saves it."""
    device = device or get_device()
    data = data.to(device)
    loss_fn = LOSSES[loss]

    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = DdiModel(n_nodes=data.n_nodes, **model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metrics = train(
        model, optimizer, data, loss_fn, num_epochs, batch_size, generator, eval_every
    )

    test = evaluate(model, data.graph, data.splits["test"], loss_fn)
    test_rows = pd.DataFrame(
        [
            {"split": "test", "metric": m, "epoch": num_epochs - 1, "mean": v, "std": 0.0}
            for m, v in test.items()
        ]
    )
    metrics = pd.concat([metrics, test_rows], ignore_index=True)

    config = {
        "model_params": {"n_nodes": data.n_nodes, **model_params},
        "loss": loss,
        "lr": lr,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "eval_every": eval_every,
        "seed": seed,
    }
    save_run(Path(save_dir) / name, model, optimizer, metrics, config)
    return model, metrics


def save_run(run_dir: Path, model, optimizer, metrics: pd.DataFrame, config: dict) -> None:
    """Saves the weights, the settings and the metric history of a run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        run_dir / "checkpoint.pt",
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    metrics.to_csv(run_dir / "metrics.csv", index=False)


def load_model(run_dir: str, device: torch.device | None = None) -> DdiModel:
    """Rebuilds a saved model and loads its weights."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    device = device or get_device()

    model = DdiModel(**config["model_params"]).to(device)
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()
