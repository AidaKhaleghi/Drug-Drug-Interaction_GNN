import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from ddi_gnn.data import DdiData
from ddi_gnn.train import run_experiment

SPLITS = ("train", "valid", "test")


def run_seeds(
    name: str, data: DdiData, model_params: dict, seeds: list[int], **kwargs
) -> pd.DataFrame:
    """Trains the same setting once per seed."""
    rows = []
    for seed in seeds:
        result = run_experiment(
            f"{name}/seed-{seed}", data, model_params, seed=seed, **kwargs
        )
        rows.append(
            {
                "config": name,
                "seed": seed,
                "best_epoch": result.best_epoch,
                **result.summary,
            }
        )
    return pd.DataFrame(rows)


def run_sweep(
    prefix: str,
    data: DdiData,
    base_params: dict,
    candidates: list[dict],
    seeds: list[int],
    **kwargs,
) -> pd.DataFrame:
    """Runs every candidate with all seeds."""
    runs = []
    for candidate in tqdm(candidates, desc=prefix):
        name = f"{prefix}/" + "_".join(f"{k}-{v}" for k, v in candidate.items())
        result = run_seeds(name, data, {**base_params, **candidate}, seeds, **kwargs)
        runs.append(result.assign(**candidate))
    return pd.concat(runs, ignore_index=True)


def summarize_runs(runs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean and std over seeds. Columns: <by>, then <split>_mean and <split>_std."""
    columns = [c for c in (*SPLITS, "best_epoch") if c in runs]
    summary = runs.groupby(by, sort=False)[columns].agg(["mean", "std"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    return summary.reset_index()


def best_config(summary: pd.DataFrame, keys: list[str]) -> dict:
    """Returns the values of `keys` in the row with the highest mean validation Hits@20."""
    best = summary["valid_mean"].idxmax()
    # Read each column on its own: a whole row would turn ints and bools into floats.
    return {key: _to_python(summary.loc[best, key]) for key in keys}


def mean_std(row: pd.Series, split: str, digits: int = 3) -> str:
    """Formats the mean and std of a split from a summary row."""
    mean, std = row[f"{split}_mean"], row[f"{split}_std"]
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def format_summary(summary: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Turns a summary into a readable table: `keys`, then 'mean ± std' per split."""
    table = summary[keys].copy()
    for split in SPLITS:
        if f"{split}_mean" in summary:
            table[split] = summary.apply(lambda row: mean_std(row, split), axis=1)
    table["best epoch"] = summary["best_epoch_mean"].round().astype(int)
    return table


def load_history(save_dir: str, name: str) -> tuple[pd.DataFrame, int]:
    """Reads the metric history and the best epoch of a saved run."""
    run_dir = Path(save_dir) / name
    config = json.loads((run_dir / "config.json").read_text())
    return pd.read_csv(run_dir / "metrics.csv"), config["best_epoch"]


def _to_python(value):
    """Turns NumPy scalars into plain Python values (they are saved as JSON)."""
    return value.item() if hasattr(value, "item") else value
