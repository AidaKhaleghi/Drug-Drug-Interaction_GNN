from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from ogb.linkproppred import LinkPropPredDataset


@dataclass
class Graph:
    """An undirected graph stored as a directed edge list with both directions."""

    edge_index: torch.Tensor  # Shape (2, n_edges): senders, then receivers
    num_nodes: int

    def to(self, device: torch.device) -> Graph:
        return Graph(self.edge_index.to(device), self.num_nodes)


@dataclass
class Pairs:
    """Positive (known interaction) and negative (no known interaction) drug pairs."""

    pos: torch.Tensor  # Shape (n_pos, 2)
    neg: torch.Tensor  # Shape (n_neg, 2)

    def to(self, device: torch.device) -> Pairs:
        return Pairs(self.pos.to(device), self.neg.to(device))

    def train_batches(
        self, batch_size: int, generator: torch.Generator
    ) -> Iterator[Pairs]:
        """Yields shuffled batches with one fresh random negative per positive."""
        n_pos = self.pos.shape[0]
        if self.neg.shape[0] < n_pos:
            raise ValueError("Need at least as many negative pairs as positive pairs.")

        device = self.pos.device
        pos_order = torch.randperm(n_pos, generator=generator).to(device)
        neg_order = torch.randperm(self.neg.shape[0], generator=generator)[:n_pos]
        neg_order = neg_order.to(device)

        for start in range(0, n_pos, batch_size):
            end = start + batch_size
            yield Pairs(self.pos[pos_order[start:end]], self.neg[neg_order[start:end]])


@dataclass
class DdiData:
    """The training graph, the pairs of each split, and drug names for each node."""

    graph: Graph
    splits: dict[str, Pairs]
    annotation: pd.DataFrame

    @property
    def n_nodes(self) -> int:
        return self.graph.num_nodes

    def to(self, device: torch.device) -> DdiData:
        return DdiData(
            graph=self.graph.to(device),
            splits={name: pairs.to(device) for name, pairs in self.splits.items()},
            annotation=self.annotation,
        )


def load_ddi(
    root: str = "datasets",
    node_limit: int | None = None,
    generator: torch.Generator | None = None,
) -> DdiData:
    """Downloads (if needed) and prepares ogbl-ddi."""
    n_nodes, split_edge = _download(root)
    pos_edges = {
        name: torch.as_tensor(split_edge[name]["edge"], dtype=torch.long)
        for name in ("train", "valid", "test")
    }
    neg_edges = {
        name: torch.as_tensor(split_edge[name]["edge_neg"], dtype=torch.long)
        for name in ("valid", "test")
    }
    return build_ddi_data(
        n_nodes, pos_edges, neg_edges, _load_annotation(root), node_limit, generator
    )


def build_ddi_data(
    n_nodes: int,
    pos_edges: dict[str, torch.Tensor],
    neg_edges: dict[str, torch.Tensor],
    annotation: pd.DataFrame,
    node_limit: int | None = None,
    generator: torch.Generator | None = None,
) -> DdiData:
    """Builds DdiData from raw edge tensors."""
    if node_limit is not None and node_limit < n_nodes:
        if generator is None:
            raise ValueError("A generator is required when node_limit is set.")
        node_ids = torch.randperm(n_nodes, generator=generator)[:node_limit]
        node_ids = node_ids.sort().values

        # mapping[old_id] = new_id. Drugs outside the subset get -1.
        mapping = torch.full((n_nodes,), -1, dtype=torch.long)
        mapping[node_ids] = torch.arange(len(node_ids))

        pos_edges = {k: _remap_pairs(v, mapping) for k, v in pos_edges.items()}
        neg_edges = {k: _remap_pairs(v, mapping) for k, v in neg_edges.items()}
        annotation = _remap_annotation(annotation, mapping)
        n_nodes = len(node_ids)

    graph = _make_graph(n_nodes, pos_edges["train"])
    splits = {"train": Pairs(pos_edges["train"], _infer_negative_pairs(graph))}
    for name in ("valid", "test"):
        splits[name] = Pairs(pos_edges[name], neg_edges[name])
    return DdiData(graph, splits, annotation)


@contextmanager
def _allow_pickled_torch_load() -> Iterator[None]:
    """Lets OGB read its own files with newer PyTorch versions.

    PyTorch 2.6+ loads with weights_only=True by default. OGB files contain
    NumPy arrays, so loading fails. Only use this for the official OGB files.
    """
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load


def _download(root: str) -> tuple[int, dict]:
    with _allow_pickled_torch_load():
        raw = LinkPropPredDataset(name="ogbl-ddi", root=root)
        split_edge = raw.get_edge_split()
    return int(raw[0]["num_nodes"]), split_edge


def _load_annotation(root: str) -> pd.DataFrame:
    """Maps each node id to its DrugBank id and drug name."""
    mapping_dir = Path(root) / "ogbl_ddi" / "mapping"
    descriptions = pd.read_csv(mapping_dir / "ddi_description.csv.gz")
    node_to_dbid = pd.read_csv(mapping_dir / "nodeidx2drugid.csv.gz").rename(
        columns={"drug id": "dbid", "node idx": "node_id"}
    )

    # Drug names appear in two columns (first and second drug of each pair).
    names = pd.concat(
        [
            descriptions[["first drug id", "first drug name"]].set_axis(
                ["dbid", "drug_name"], axis=1
            ),
            descriptions[["second drug id", "second drug name"]].set_axis(
                ["dbid", "drug_name"], axis=1
            ),
        ]
    ).drop_duplicates(subset="dbid")

    return node_to_dbid.merge(names, on="dbid", how="left")


def _make_graph(n_nodes: int, pos_edges: torch.Tensor) -> Graph:
    """Builds an undirected graph by adding each edge in both directions."""
    senders, receivers = pos_edges[:, 0], pos_edges[:, 1]
    edge_index = torch.stack(
        [torch.cat([senders, receivers]), torch.cat([receivers, senders])]
    )
    return Graph(edge_index, n_nodes)


def _infer_negative_pairs(graph: Graph) -> torch.Tensor:
    """Returns every pair (i < j) that is not an edge of the graph."""
    n = graph.num_nodes
    is_negative = torch.ones((n, n), dtype=torch.bool)
    senders, receivers = graph.edge_index
    is_negative[senders, receivers] = False
    return torch.triu(is_negative, diagonal=1).nonzero()


def _remap_pairs(pairs: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Keeps pairs with both drugs in the subset and renumbers the drugs."""
    keep = (mapping[pairs[:, 0]] >= 0) & (mapping[pairs[:, 1]] >= 0)
    return mapping[pairs[keep]]


def _remap_annotation(annotation: pd.DataFrame, mapping: torch.Tensor) -> pd.DataFrame:
    """Keeps annotation rows of drugs in the subset and renumbers node ids."""
    new_ids = mapping.numpy()[annotation["node_id"].to_numpy()]
    keep = new_ids >= 0
    return annotation[keep].assign(node_id=new_ids[keep]).reset_index(drop=True)
