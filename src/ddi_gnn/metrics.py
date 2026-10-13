from collections import defaultdict

import pandas as pd
import torch


@torch.no_grad()
def hits_at_k(scores: dict[str, torch.Tensor], k: int = 20) -> float:
    """Fraction of positive pairs that score higher than the k-th best negative."""
    pos, neg = scores["pos"], scores["neg"]
    if neg.numel() < k:
        raise ValueError(f"Hits@{k} needs at least {k} negative pairs.")
    threshold = torch.topk(neg, k).values[-1]
    return (pos > threshold).float().mean().item()


class MetricsLogger:
    """Collects metric values per step and stores their mean and std per epoch."""

    def __init__(self):
        self._buffer = defaultdict(lambda: defaultdict(list))
        self._rows = []
        self._latest = {}

    def log(self, split: str, metrics: dict[str, float]) -> None:
        """Stores the metrics of one step."""
        for metric, value in metrics.items():
            self._buffer[split][metric].append(value)

    def flush(self, epoch: int) -> None:
        """Averages the logged steps into one row per split and metric."""
        for split, metrics in self._buffer.items():
            for metric, values in metrics.items():
                mean = sum(values) / len(values)
                std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
                self._rows.append(
                    {
                        "split": split,
                        "metric": metric,
                        "epoch": epoch,
                        "mean": mean,
                        "std": std,
                    }
                )
                self._latest[(split, metric)] = mean
        self._buffer.clear()

    def latest(self, metric: str) -> str:
        """Returns the last value of a metric for each split, as text."""
        parts = [
            f"{split}={value:.4f}"
            for (split, name), value in self._latest.items()
            if name == metric
        ]
        return f"{metric}: " + ", ".join(parts)

    def to_frame(self) -> pd.DataFrame:
        """Returns all logged values. Columns: split, metric, epoch, mean, std."""
        return pd.DataFrame(self._rows)
