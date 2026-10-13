import torch
import torch.nn as nn
import torch.nn.functional as F

from ddi_gnn.data import Graph, Pairs


class SAGEConv(nn.Module):
    """GraphSAGE layer: aggregate neighbors, concatenate with self, then project."""

    def __init__(self, embedding_dim: int, with_self: bool, degree_norm: bool):
        super().__init__()
        self.with_self = with_self
        self.degree_norm = degree_norm
        self.dense = nn.Linear(2 * embedding_dim, embedding_dim)

    def forward(self, graph: Graph, x: torch.Tensor) -> torch.Tensor:
        n_nodes = x.size(0)
        senders, receivers = graph.edge_index

        if self.with_self:
            self_ids = torch.arange(n_nodes, device=x.device)
            senders = torch.cat([senders, self_ids])
            receivers = torch.cat([receivers, self_ids])

        degree = torch.bincount(receivers, minlength=n_nodes).clamp(min=1).float()

        messages = x[senders]
        if self.degree_norm:
            weights = torch.rsqrt(degree[senders] * degree[receivers])
            messages = messages * weights[:, None]

        aggregated = torch.zeros_like(x).index_add_(0, receivers, messages)
        if not self.degree_norm:
            aggregated = aggregated / degree[:, None]

        return self.dense(torch.cat([x, aggregated], dim=-1))


class NodeEncoder(nn.Module):
    """Learns one embedding per drug and refines it with two SAGE layers."""

    def __init__(
        self,
        n_nodes: int,
        embedding_dim: int,
        last_layer_self: bool,
        degree_norm: bool,
        dropout_rate: float,
    ):
        super().__init__()
        self.node_embeddings = nn.Embedding(n_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)

        self.conv1 = SAGEConv(embedding_dim, with_self=True, degree_norm=degree_norm)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = SAGEConv(
            embedding_dim, with_self=last_layer_self, degree_norm=degree_norm
        )

    def forward(self, graph: Graph) -> torch.Tensor:
        x = self.node_embeddings.weight
        if graph.num_nodes != x.size(0):
            raise ValueError("The graph has a different number of nodes than the model.")

        x = F.relu(self.conv1(graph, x))
        x = self.dropout(x)
        return self.conv2(graph, x)


class LinkPredictor(nn.Module):
    """Scores a drug pair from the element-wise product of its two embeddings."""

    def __init__(self, embedding_dim: int, n_layers: int, dropout_rate: float):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim) for _ in range(n_layers - 1)]
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(
        self, sender_embeddings: torch.Tensor, receiver_embeddings: torch.Tensor
    ) -> torch.Tensor:
        x = sender_embeddings * receiver_embeddings
        for layer in self.hidden_layers:
            x = self.dropout(F.relu(layer(x)))
        return self.output_layer(x).squeeze(-1)


class DdiModel(nn.Module):
    """Predicts drug-drug interactions: node encoder followed by link predictor."""

    def __init__(
        self,
        n_nodes: int,
        embedding_dim: int,
        dropout_rate: float,
        last_layer_self: bool,
        degree_norm: bool,
        n_mlp_layers: int = 2,
    ):
        super().__init__()
        self.node_encoder = NodeEncoder(
            n_nodes, embedding_dim, last_layer_self, degree_norm, dropout_rate
        )
        self.link_predictor = LinkPredictor(embedding_dim, n_mlp_layers, dropout_rate)

    def encode(self, graph: Graph) -> torch.Tensor:
        """Returns one embedding per node."""
        return self.node_encoder(graph)

    def score(self, embeddings: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """Returns one score (logit) per drug pair. Shape of pairs: (n, 2)."""
        return self.link_predictor(embeddings[pairs[:, 0]], embeddings[pairs[:, 1]])

    def forward(self, graph: Graph, pairs: Pairs) -> dict[str, torch.Tensor]:
        embeddings = self.encode(graph)
        return {
            "pos": self.score(embeddings, pairs.pos),
            "neg": self.score(embeddings, pairs.neg),
        }
