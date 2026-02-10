import pickle
from typing import cast, overload, Literal, Mapping, NamedTuple, Optional, Tuple, Union
from functools import partial

from pgx import Env
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph

from jpyger import state_to_graph
import chess_graph as cg
from chess_graph import HeteroGraph, EdgeSet
from models_deprecated import EdgeNet


class AttentionPooling(nn.Module):
    @nn.compact
    def __call__(
        self,
        *args,
        x: jnp.ndarray,
        segment_ids: jnp.ndarray,
        mask: Optional[jnp.ndarray]=None,
        num_segments: Optional[int]=None,
        **kwargs
    ):
        if mask is None:
            segment_ids_masked = segment_ids
        else:
            segment_ids_masked = jnp.where(mask, segment_ids, -1)
        att = cast(jnp.ndarray, jraph.segment_softmax(
            nn.Dense(1)(x).squeeze(-1),
            segment_ids_masked,
            num_segments
        ))
        if mask is not None:
            att = att * mask
        att = jnp.tile(att, (x.shape[1], 1)).transpose()
        return jraph.segment_sum(x * att, segment_ids, num_segments)


class BNR(nn.Module):
    momentum: float = 0.9
    @nn.compact
    def __call__(
        self,
        *args,
        x,
        training: bool=False,
        **kwargs
    ):
        training=False
        x = nn.BatchNorm(momentum=self.momentum)(x, use_running_average=not training)
        x = jax.nn.relu(x)
        return x


class GATEAU(nn.Module):
    out_dim: int = 128
    mix_edge_node: bool = False
    add_features: bool = True
    self_edges: bool = False
    simple_update: bool = True
    sync_updates: Optional[bool] = None
    @nn.compact
    def __call__(
        self,
        *args,
        graph: jraph.GraphsTuple,
        **kwargs
    ) -> jraph.GraphsTuple:
        try:
            sum_n_node = graph.nodes.shape[0] # type: ignore
        except IndexError:
            raise IndexError('GAT requires node features')

        if self.sync_updates is None:
            sync_updates = not self.simple_update
        else:
            sync_updates = self.sync_updates

        node_features = cast(jnp.ndarray, graph.nodes)
        edge_features = cast(jnp.ndarray, graph.edges)

        sent_attributes_1 = nn.Dense(self.out_dim)(node_features)[graph.senders]
        if self.simple_update:
            sent_attributes_2 = node_features[graph.senders]
        else:
            sent_attributes_2 = nn.Dense(self.out_dim)(node_features)[graph.senders]
        received_attributes = nn.Dense(self.out_dim)(
            node_features
        )[graph.receivers]
        if sync_updates:
            edge_features_0 = nn.Dense(self.out_dim)(edge_features)
        else:
            edge_features_0 = None
        edge_features = nn.Dense(self.out_dim)(edge_features)

        if self.add_features:
            edge_features = (
                  sent_attributes_1
                + edge_features
                + received_attributes
            )
        else:
            edge_features = (
                  sent_attributes_1
                * edge_features
                * received_attributes
            )

        attention_coeffs = nn.Dense(1)(edge_features)
        attention_coeffs = nn.leaky_relu(attention_coeffs)
        attention_weights = jraph.segment_softmax(
            attention_coeffs,
            segment_ids=cast(jnp.ndarray, graph.receivers),
            num_segments=sum_n_node
        )

        if self.mix_edge_node:
            if self.add_features:
                message = sent_attributes_2 + (
                    edge_features_0 if sync_updates else edge_features
                )
            else:
                message = sent_attributes_2 * (
                    edge_features_0 if sync_updates else edge_features
                )
        else:
            message = sent_attributes_2
        if self.simple_update:
            message = nn.Dense(self.out_dim)(message)
        message = attention_weights * message
        node_features = jraph.segment_sum(
            message,
            segment_ids=cast(jnp.ndarray, graph.receivers),
            num_segments=sum_n_node
        )
        if self.self_edges:
            node_features += nn.Dense(self.out_dim)(graph.nodes)

        return graph._replace(
            nodes=node_features,
            edges=edge_features
        )


class EGNN3(nn.Module):
    out_dim: int = 128
    mix_edge_node: bool = False
    add_features: bool = True
    self_edges: bool = False
    simple_update: bool = True
    sync_updates: Optional[bool] = None
    @nn.compact
    def __call__(
        self,
        *args,
        graph: jraph.GraphsTuple,
        training: bool=False,
        **kwargs
    ) -> jraph.GraphsTuple:
        i, j = map(partial(cast, jraph.ArrayTree), (graph.nodes, graph.edges))
        graph = GATEAU(
            out_dim=self.out_dim,
            mix_edge_node=self.mix_edge_node,
            add_features=self.add_features,
            self_edges=self.self_edges,
            simple_update=self.simple_update,
            sync_updates=self.sync_updates
        )(
            graph=graph._replace(
                nodes=BNR()(x=graph.nodes, training=training),
                edges=BNR()(x=graph.edges, training=training)
            )
        )
        graph = GATEAU(
            out_dim=self.out_dim,
            mix_edge_node=self.mix_edge_node,
            add_features=self.add_features,
            self_edges=self.self_edges,
            simple_update=self.simple_update,
            sync_updates=self.sync_updates
        )(
            graph=graph._replace(
                nodes=BNR()(x=graph.nodes, training=training),
                edges=BNR()(x=graph.edges, training=training)
            )
        )
        return graph._replace(nodes=graph.nodes+i, edges=graph.edges+j)


# ---------------------------------------------------------------------------
# Heterogeneous multi-graph modules
# ---------------------------------------------------------------------------

# Edge type names matching HeteroGraph fields (order matters)
_EDGE_TYPE_NAMES = (
    'move_edges', 'grid_edges', 'attack_edges', 'defense_edges',
    'file_edges', 'rank_edges', 'diagonal_edges',
)


class HeteroGATEAU(nn.Module):
    """Run separate GATEAU per edge type, sum node updates."""
    out_dim: int = 128
    mix_edge_node: bool = False
    add_features: bool = True
    self_edges: bool = False
    simple_update: bool = True
    sync_updates: Optional[bool] = None

    @nn.compact
    def __call__(
        self,
        *args,
        nodes: jnp.ndarray,
        edge_sets: Tuple[EdgeSet, ...],
        training: bool = False,
        **kwargs
    ) -> Tuple[jnp.ndarray, Tuple[EdgeSet, ...]]:
        """
        Args:
            nodes: (total_nodes, dim) — shared node features
            edge_sets: tuple of 7 EdgeSets (one per edge type)
            training: batch norm mode
        Returns:
            (new_nodes, new_edge_sets)
        """
        sum_n_node = nodes.shape[0]
        # Shared BNR on nodes (applied once, not per edge type)
        bnr_nodes = BNR()(x=nodes, training=training)

        node_updates = jnp.zeros_like(bnr_nodes)
        new_edge_sets = []

        for i, es in enumerate(edge_sets):
            # Per-edge-type BNR on edge features
            bnr_ef = BNR(name=f"bnr_edge_{i}")(x=es.features, training=training)

            # Build a temporary GraphsTuple for GATEAU
            tmp_graph = jraph.GraphsTuple(
                n_node=jnp.array([sum_n_node]),
                nodes=bnr_nodes,
                n_edge=jnp.array([es.senders.shape[0]]),
                edges=bnr_ef,
                senders=es.senders,
                receivers=es.receivers,
                globals=None,
            )
            out_graph = GATEAU(
                out_dim=self.out_dim,
                mix_edge_node=self.mix_edge_node,
                add_features=self.add_features,
                self_edges=self.self_edges,
                simple_update=self.simple_update,
                sync_updates=self.sync_updates,
                name=f"gateau_{i}",
            )(graph=tmp_graph)

            node_updates = node_updates + cast(jnp.ndarray, out_graph.nodes)
            new_edge_sets.append(EdgeSet(
                senders=es.senders,
                receivers=es.receivers,
                features=cast(jnp.ndarray, out_graph.edges),
            ))

        return node_updates, tuple(new_edge_sets)


class HeteroEGNN(nn.Module):
    """Residual block: 2x HeteroGATEAU + residual on nodes and per-type edges."""
    out_dim: int = 128
    mix_edge_node: bool = False
    add_features: bool = True
    self_edges: bool = False
    simple_update: bool = True
    sync_updates: Optional[bool] = None

    @nn.compact
    def __call__(
        self,
        *args,
        nodes: jnp.ndarray,
        edge_sets: Tuple[EdgeSet, ...],
        training: bool = False,
        **kwargs
    ) -> Tuple[jnp.ndarray, Tuple[EdgeSet, ...]]:
        # Save residuals
        res_nodes = nodes
        res_edges = tuple(es.features for es in edge_sets)

        # First HeteroGATEAU
        nodes, edge_sets = HeteroGATEAU(
            out_dim=self.out_dim,
            mix_edge_node=self.mix_edge_node,
            add_features=self.add_features,
            self_edges=self.self_edges,
            simple_update=self.simple_update,
            sync_updates=self.sync_updates,
        )(nodes=nodes, edge_sets=edge_sets, training=training)

        # Second HeteroGATEAU
        nodes, edge_sets = HeteroGATEAU(
            out_dim=self.out_dim,
            mix_edge_node=self.mix_edge_node,
            add_features=self.add_features,
            self_edges=self.self_edges,
            simple_update=self.simple_update,
            sync_updates=self.sync_updates,
        )(nodes=nodes, edge_sets=edge_sets, training=training)

        # Residual connections
        nodes = nodes + res_nodes
        edge_sets = tuple(
            EdgeSet(
                senders=es.senders,
                receivers=es.receivers,
                features=es.features + rf,
            )
            for es, rf in zip(edge_sets, res_edges)
        )

        return nodes, edge_sets


class HeteroEdgeNet(nn.Module):
    """Full heterogeneous multi-graph model."""
    n_actions: int
    inner_size: int = 128
    n_res_layers: int = 5
    attention_pooling: bool = True
    mix_edge_node: bool = False
    add_features: bool = True
    self_edges: bool = False
    simple_update: bool = True
    sync_updates: Optional[bool] = None

    @nn.compact
    def __call__(
        self,
        *args,
        graphs: HeteroGraph,
        training: bool = False,
        **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # 1. Embed nodes
        nodes = nn.Dense(self.inner_size)(graphs.nodes)

        # 2. Embed each edge type with separate Dense
        edge_sets = []
        raw_edge_sets = (
            graphs.move_edges, graphs.grid_edges,
            graphs.attack_edges, graphs.defense_edges,
            graphs.file_edges, graphs.rank_edges, graphs.diagonal_edges,
        )
        for i, es in enumerate(raw_edge_sets):
            emb_feats = nn.Dense(self.inner_size, name=f"edge_embed_{i}")(es.features)
            edge_sets.append(EdgeSet(
                senders=es.senders,
                receivers=es.receivers,
                features=emb_feats,
            ))
        edge_sets = tuple(edge_sets)

        # 3. N residual blocks
        for _ in range(self.n_res_layers):
            nodes, edge_sets = HeteroEGNN(
                out_dim=self.inner_size,
                mix_edge_node=self.mix_edge_node,
                add_features=self.add_features,
                self_edges=self.self_edges,
                simple_update=self.simple_update,
                sync_updates=self.sync_updates,
            )(nodes=nodes, edge_sets=edge_sets, training=training)

        # 4. Policy head — uses MOVE edges (index 0)
        x = BNR()(x=nodes, training=training)
        move_feats = edge_sets[0].features
        y = BNR()(x=move_feats, training=training)
        logits = nn.Dense(self.inner_size)(y)
        logits = BNR()(x=logits, training=training)
        logits = nn.Dense(1)(logits).squeeze()
        logits = logits[graphs.globals]

        # 5. Value head
        n_partitions = len(graphs.n_node)
        segment_ids = jnp.repeat(
            jnp.arange(n_partitions),
            graphs.n_node,
            axis=0,
            total_repeat_length=x.shape[0]
        )
        v = nn.Dense(self.inner_size)(x)
        v = nn.BatchNorm(momentum=0.9)(v, use_running_average=not training)
        v = jax.nn.relu(v)
        if self.attention_pooling:
            v = AttentionPooling()(
                x=v,
                segment_ids=segment_ids,
                num_segments=graphs.n_node.shape[0]
            )
        v = jax.nn.relu(v)
        v = nn.Dense(1)(v)
        v = nn.tanh(v)

        return logits, v


# AlphaGateau (deprecated)
class EdgeNet2(nn.Module):
    n_actions: int
    inner_size: int = 128
    n_res_layers: int = 5
    attention_pooling: bool = True
    mix_edge_node: bool = False
    add_features: bool = True
    self_edges: bool = False
    simple_update: bool = True
    sync_updates: Optional[bool] = None

    dot_v2: bool = True
    use_embedding: bool = True

    @nn.compact
    def __call__(
        self,
        *args,
        graphs: jraph.GraphsTuple,
        training: bool=False,
        **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        graphs = graphs._replace(
            nodes=nn.Dense(self.inner_size)(graphs.nodes),
            edges=nn.Dense(self.inner_size)(graphs.edges)
        )

        for _ in range(self.n_res_layers):
            graphs = EGNN3(
                out_dim=self.inner_size,
                mix_edge_node=self.mix_edge_node,
                add_features=self.add_features,
                self_edges=self.self_edges,
                simple_update=self.simple_update,
                sync_updates=self.sync_updates
            )(
                graph=graphs,
                training=training
            )

        # TODO: merge node and edge features from all layers
        x = BNR()(x=graphs.nodes, training=training) # type: ignore
        y = BNR()(x=graphs.edges, training=training)

        logits = nn.Dense(self.inner_size)(y)
        logits = BNR()(x=logits, training=training)
        logits = nn.Dense(1)(logits).squeeze()
        logits = logits[graphs.globals]

        n_partitions = len(graphs.n_node)
        segment_ids = jnp.repeat(
            jnp.arange(n_partitions),
            graphs.n_node,
            axis=0,
            total_repeat_length=x.shape[0]
        )
        # v = BNR()(x=x, training=training)
        v = nn.Dense(self.inner_size)(x)
        v = nn.BatchNorm(momentum=0.9)(v, use_running_average=not training)
        v = jax.nn.relu(v)
        if self.attention_pooling:
            v = AttentionPooling()(
                x=v,
                segment_ids=segment_ids,
                num_segments=graphs.n_node.shape[0]
            )
        else:
            raise DeprecationWarning
            # Mean Pooling
            # v = v * jnp.tile(node_mask, (self.inner_size, 1)).transpose()
            # v = jraph.segment_sum(v, segment_ids, graphs.n_node.shape[0])
            # v /= jnp.tile(graphs.n_node - 1, (self.inner_size, 1)).transpose()
        v = v
        v = jax.nn.relu(v) # Probably useless after attention pooling
        v = nn.Dense(1)(v)
        v = nn.tanh(v)

        return logits, v


# AlphaGateau
class AlphaGateau(nn.Module):
    n_actions: int
    inner_size: int = 128
    n_res_layers: int = 5
    attention_pooling: bool = True
    mix_edge_node: bool = False
    add_features: bool = True
    self_edges: bool = False
    simple_update: bool = True
    sync_updates: Optional[bool] = None

    dot_v2: bool = True
    use_embedding: bool = True

    @nn.compact
    def __call__(
        self,
        *args,
        graphs: jraph.GraphsTuple,
        training: bool=False,
        **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        graphs = graphs._replace(
            nodes=nn.Dense(self.inner_size)(graphs.nodes),
            edges=nn.Dense(self.inner_size)(graphs.edges)
        )

        for _ in range(self.n_res_layers):
            graphs = EGNN3(
                out_dim=self.inner_size,
                mix_edge_node=self.mix_edge_node,
                add_features=self.add_features,
                self_edges=self.self_edges,
                simple_update=self.simple_update,
                sync_updates=self.sync_updates
            )(
                graph=graphs,
                training=training
            )

        # TODO: merge node and edge features from all layers
        x = BNR()(x=graphs.nodes, training=training) # type: ignore
        y = BNR()(x=graphs.edges, training=training)

        logits = nn.Dense(self.inner_size)(y)
        logits = BNR()(x=logits, training=training)
        logits = nn.Dense(1)(logits).squeeze()
        logits = logits[graphs.globals]

        n_partitions = len(graphs.n_node)
        segment_ids = jnp.repeat(
            jnp.arange(n_partitions),
            graphs.n_node,
            axis=0,
            total_repeat_length=x.shape[0]
        )
        v = BNR()(x=x, training=training)
        v = nn.Dense(self.inner_size)(x)
        v = BNR()(x=x, training=training)
        if self.attention_pooling:
            v = AttentionPooling()(
                x=v,
                segment_ids=segment_ids,
                num_segments=graphs.n_node.shape[0]
            )
        else:
            raise DeprecationWarning
            # Mean Pooling
            # v = v * jnp.tile(node_mask, (self.inner_size, 1)).transpose()
            # v = jraph.segment_sum(v, segment_ids, graphs.n_node.shape[0])
            # v /= jnp.tile(graphs.n_node - 1, (self.inner_size, 1)).transpose()
        v = jax.nn.relu(v) # Probably useless after attention pooling
        v = nn.Dense(1)(v)
        v = nn.tanh(v)

        return logits, v


class BlockV2(nn.Module):
    num_channels: int
    name: Optional[str] = "BlockV2"

    @nn.compact
    def __call__(self, *args, x, training, **kwargs):
        i = x
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
        x = jax.nn.relu(x)
        x = nn.Conv(self.num_channels, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
        x = jax.nn.relu(x)
        x = nn.Conv(self.num_channels, kernel_size=(3, 3))(x)
        return x + i


# AlphaZero taken from mctx
class AZNet(nn.Module):
    """AlphaZero NN architecture."""
    n_actions: int
    inner_size: int = 64
    n_res_layers: int = 5 # num_blocks
    resnet_v2: bool = True
    resnet_cls = BlockV2
    name: Optional[str] = "az_net"
    # Useless parameters
    dot_v2: bool = True
    use_embedding: bool = True
    attention_pooling: bool = True
    mix_edge_node: bool = False
    add_features: bool = True
    self_edges: bool = False
    simple_update: bool = True
    sync_updates: Optional[bool] = None


    @nn.compact
    def __call__(self, *args, x, training=False, **kwargs):
        x = x.astype(jnp.float32)
        x = nn.Conv(self.inner_size, kernel_size=(3, 3))(x)

        if not self.resnet_v2:
            x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
            x = jax.nn.relu(x)

        for i in range(self.n_res_layers):
            x = self.resnet_cls(num_channels=self.inner_size, name=f"block_{i}")(
                x=x, training=training
            )

        if self.resnet_v2:
            x = nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)
            x = jax.nn.relu(x)

        # policy head
        logits = nn.Conv(features=2, kernel_size=(1, 1))(x)
        logits = nn.BatchNorm(momentum=0.9)(logits, use_running_average=not training)
        logits = jax.nn.relu(logits)
        logits = logits.reshape((logits.shape[0], -1))
        logits = nn.Dense(self.n_actions)(logits)

        # value head
        v = nn.Conv(features=1, kernel_size=(1, 1))(x)
        v = nn.BatchNorm(momentum=0.9)(v, use_running_average=not training)
        v = jax.nn.relu(v)
        v = v.reshape((v.shape[0], -1))
        v = nn.Dense(self.inner_size)(v)
        v = jax.nn.relu(v)
        v = nn.Dense(1)(v)
        v = jnp.tanh(v)
        v = v.reshape((-1,))

        return logits, v


state_to_graph = jax.jit(state_to_graph, static_argnames='use_embedding')
new_state_to_graph = jax.jit(cg.state_to_graph)
hetero_state_to_graph = jax.jit(cg.state_to_hetero_graph)


class ModelManager(NamedTuple):
    id: str
    model: nn.Module
    use_embedding: bool = True
    use_graph: bool = True
    new_graph: bool = True
    hetero_graph: bool = False

    def init(self, key: chex.PRNGKey, x):
        if self.use_graph:
            return self.model.init(key, graphs=x)
        return self.model.init(key, x=x)

    @overload
    def __call__(
        self,
        x,
        legal_action_mask: jnp.ndarray,
        params: chex.ArrayTree,
        training: Literal[False]=False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ...

    @overload
    def __call__(
        self,
        x,
        legal_action_mask: jnp.ndarray,
        params: chex.ArrayTree,
        training: Literal[True]
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], chex.ArrayTree]:
        ...

    def __call__(
        self,
        x,
        legal_action_mask: jnp.ndarray,
        params: chex.ArrayTree,
        training: bool=False
    ) -> Union[
        Tuple[jnp.ndarray, jnp.ndarray],
        Tuple[Tuple[jnp.ndarray, jnp.ndarray], chex.ArrayTree]
    ]:
        if self.use_graph:
            r_tuple, batch_stats = self.model.apply(
                cast(Mapping, params),
                graphs=x,
                mutable=['batch_stats'],
                training=training
            )
        else:
            r_tuple, batch_stats = self.model.apply(
                cast(Mapping, params),
                x=x,
                mutable=['batch_stats'],
                training=training
            )
        logits, value = r_tuple
        value = jnp.reshape(value, (-1,))
        logits = logits.reshape((value.shape[-1], -1))

        # mask invalid actions
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(
            legal_action_mask,
            logits,
            jnp.finfo(logits.dtype).min
        )
        if training:
            return (logits, value), batch_stats['batch_stats']
        return logits, value

    def format_data(self, state=None, board=None, observation=None,
            legal_action_mask=None, **kwargs):
        if self.use_graph:
            if state is not None:
                board = state._board
                observation = state.observation
                legal_action_mask = state.legal_action_mask
            if self.hetero_graph:
                return hetero_state_to_graph(
                    observation, legal_action_mask,
                )
            if self.new_graph:
                return new_state_to_graph(
                    observation, legal_action_mask,
                )
            return state_to_graph(
                board, observation, legal_action_mask,
                use_embedding=self.use_embedding
            )
        return state.observation if state is not None else observation

def load_model(
    env: Env,
    file_name: str,
    model_name: str,
):
    with open(file_name, "rb") as f:
        dic = pickle.load(f)
        net = AZNet
        if dic['config']['use_gnn']:
            if dic['config'].get('hetero_graph', False):
                net = HeteroEdgeNet
            elif dic['config'].get('new_graph', False):
                net = EdgeNet2
            else:
                net = EdgeNet
        model = ModelManager(
            id=model_name,
            model=net(
                n_actions=env.num_actions,
                inner_size=dic['config']['inner_size'],
                n_res_layers=dic['config'].get('n_gnn_layers', 5),
                dot_v2=dic['config'].get('dotv2', True),
                use_embedding=dic['config']['use_embedding'],
                attention_pooling=dic['config'].get('attention_pooling', True),
                mix_edge_node=dic['config'].get('mix_edge_node', False),
                add_features=dic['config'].get('add_features', True),
                self_edges=dic['config'].get('self_edges', False),
                simple_update=dic['config'].get('simple_update', True),
                sync_updates=dic['config'].get('sync_updates', None),
            ),
            use_embedding=dic['config']['use_embedding'],
            use_graph=dic['config']['use_gnn'],
            new_graph=dic['config'].get('new_graph', False),
            hetero_graph=dic['config'].get('hetero_graph', False),
        )
        model_params = {
            'params': dic['params'],
            'batch_stats': dic['batch_stats']
        }
    return model, model_params

