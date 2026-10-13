import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ddi_gnn.data import DdiData

SPLIT_COLORS = {"train": "#377eb8", "valid": "#4daf4a", "test": "#ff7f00"}


def plot_learning(
    metrics: pd.DataFrame, splits: tuple[str, ...] = ("train", "valid")
) -> plt.Figure:
    """Plots loss and Hits@20 per epoch, with a band of one std around each line."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panels = [("loss", "Loss", None), ("hits@20", "Hits@20", (0, 1))]

    for ax, (metric, title, ylim) in zip(axes, panels):
        for split in splits:
            rows = metrics[(metrics["split"] == split) & (metrics["metric"] == metric)]
            rows = rows.sort_values("epoch")
            color = SPLIT_COLORS[split]
            ax.plot(rows["epoch"], rows["mean"], label=split.title(), color=color)
            ax.fill_between(
                rows["epoch"],
                rows["mean"] - rows["std"],
                rows["mean"] + rows["std"],
                color=color,
                alpha=0.2,
                linewidth=0,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    return fig


def plot_graph(data: DdiData, figsize: tuple[int, int] = (20, 20)) -> plt.Figure:
    """Plots the training graph on a circle, with drug names as rotated labels."""
    fig = plt.figure(figsize=figsize)

    graph = nx.Graph()
    graph.add_nodes_from(range(data.n_nodes))
    graph.add_edges_from(data.graph.edge_index.T.tolist())
    pos = nx.circular_layout(graph)

    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_color="lightgray",
        edge_color="gray",
        node_size=10,
        alpha=0.1,
    )

    names = (
        data.annotation.dropna(subset=["drug_name"])
        .set_index("node_id")["drug_name"]
        .to_dict()
    )
    labels = nx.draw_networkx_labels(graph, pos, labels=names, font_size=10)

    # Rotate each label along its radius so that it stays readable.
    for node, text in labels.items():
        angle = np.degrees(np.arctan2(pos[node][1], pos[node][0]))
        if angle > 90 or angle < -90:
            angle += 180
            text.set_horizontalalignment("right")
        else:
            text.set_horizontalalignment("left")
        text.set_verticalalignment("center")
        text.set_rotation(angle)
        text.set_rotation_mode("anchor")

    return fig
