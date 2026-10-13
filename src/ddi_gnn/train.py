import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from ddi_gnn.data import DdiData, Graph, Pairs
from ddi_gnn.losses import LOSSES
from ddi_gnn.metrics import MetricsLogger, hits_at_k
from ddi_gnn.model import DdiModel
from ddi_gnn.utils import get_device, set_seed


@dataclass
class RunResult:
    """The outcome of one training run."""

    model: DdiModel  # Holds the weights of the best validation epoch
    metrics: pd.DataFrame
    best_epoch: int
    summary: dict[str, float]  # Hits@20 per split at the best epoch


def train_step(model, optimizer, graph: Graph, pairs: Pairs, loss_fn) -> dict:
    """Runs one optimizer step on a batch of pairs."""
    model.train()
    optimizer.zero_grad()

    scores = model(graph, pairs)
    loss = loss_fn(scores)
    loss.backward()
    optimizer.step()

    return {"loss": loss.item()}


@torch.no_grad()
def evaluate(model, graph: Graph, pairs: Pairs, loss_fn) -> dict:
    """Scores every pair of a split at once and returns its loss and Hits@20."""
    model.eval()
    scores = model(graph, pairs)
    return {"loss": loss_fn(scores).item(), "hits@20": hits_at_k(scores)}


def make_train_eval_pairs(data: DdiData) -> Pairs:
    """Builds the pairs used to measure Hits@20 on the training split."""
    train, valid = data.splits["train"], data.splits["valid"]
    n_neg = min(valid.neg.shape[0], train.neg.shape[0])
    generator = torch.Generator().manual_seed(0)
    chosen = torch.randperm(train.neg.shape[0], generator=generator)[:n_neg]
    return Pairs(train.pos, train.neg[chosen.to(train.neg.device)])


def train(
    model: DdiModel,
    optimizer: torch.optim.Optimizer,
    data: DdiData,
    loss_fn,
    num_epochs: int,
    batch_size: int,
    generator: torch.Generator,
    eval_every: int = 1,
) -> tuple[pd.DataFrame, int]:
    """Trains the model and returns the metric history and the best epoch."""
    logger = MetricsLogger()
    train_eval = make_train_eval_pairs(data)
    best_hits, best_epoch, best_state = -1.0, 0, None
    progress = tqdm(range(num_epochs), leave=False)

    for epoch in progress:
        progress.set_description(f"Epoch {epoch + 1}")

        for batch in data.splits["train"].train_batches(batch_size, generator):
            logger.log(
                "train_batch", train_step(model, optimizer, data.graph, batch, loss_fn)
            )

        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            valid = evaluate(model, data.graph, data.splits["valid"], loss_fn)
            logger.log("train", evaluate(model, data.graph, train_eval, loss_fn))
            logger.log("valid", valid)

            if valid["hits@20"] > best_hits:
                best_hits, best_epoch = valid["hits@20"], epoch
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }

        logger.flush(epoch)
        progress.set_postfix_str(logger.latest("hits@20"))

    model.load_state_dict(best_state)
    return logger.to_frame(), best_epoch


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
    evaluate_test: bool = False,
    device: torch.device | None = None,
) -> RunResult:
    """Trains one model from scratch and saves the weights of its best epoch."""
    device = device or get_device()
    data = data.to(device)
    loss_fn = LOSSES[loss]

    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = DdiModel(n_nodes=data.n_nodes, **model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metrics, best_epoch = train(
        model, optimizer, data, loss_fn, num_epochs, batch_size, generator, eval_every
    )

    if evaluate_test:
        test = evaluate(model, data.graph, data.splits["test"], loss_fn)
        test_rows = pd.DataFrame(
            [
                {"split": "test", "metric": m, "epoch": best_epoch, "mean": v, "std": 0.0}
                for m, v in test.items()
            ]
        )
        metrics = pd.concat([metrics, test_rows], ignore_index=True)

    hits = metrics[(metrics["metric"] == "hits@20") & (metrics["epoch"] == best_epoch)]
    summary = hits.set_index("split")["mean"].to_dict()

    config = {
        "model_params": {"n_nodes": data.n_nodes, **model_params},
        "loss": loss,
        "lr": lr,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "eval_every": eval_every,
        "seed": seed,
        "best_epoch": best_epoch,
        "evaluate_test": evaluate_test,
    }
    save_run(Path(save_dir) / name, model, metrics, config)
    return RunResult(model, metrics, best_epoch, summary)


def save_run(run_dir: Path, model, metrics: pd.DataFrame, config: dict) -> None:
    """Saves the weights, the settings and the metric history of a run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, run_dir / "checkpoint.pt")
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
