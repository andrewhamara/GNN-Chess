from typing import cast, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import jraph as jraph
import pgx
import pgx.chess as pgc
import pgx.gardner_chess as pgg
from rich.pretty import pprint


# ---------------------------------------------------------------------------
# Heterogeneous multi-graph data structures
# ---------------------------------------------------------------------------

class EdgeSet(NamedTuple):
    senders: jnp.ndarray
    receivers: jnp.ndarray
    features: jnp.ndarray


class HeteroGraph(NamedTuple):
    n_node: jnp.ndarray
    nodes: jnp.ndarray
    move_edges: EdgeSet
    grid_edges: EdgeSet
    attack_edges: EdgeSet
    defense_edges: EdgeSet
    file_edges: EdgeSet
    rank_edges: EdgeSet
    diagonal_edges: EdgeSet
    globals: jnp.ndarray  # action -> move edge mapping (for policy head)


# ---------------------------------------------------------------------------
# Static edge constructors (precomputed, board-geometry only)
# ---------------------------------------------------------------------------

def _grid_edges(size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """8-connected grid adjacency. Returns (senders, receivers, features).
    Features: (delta_file, delta_rank)."""
    senders, receivers, feats = [], [], []
    for f1 in range(size):
        for r1 in range(size):
            for f2 in range(size):
                for r2 in range(size):
                    if (f1, r1) == (f2, r2):
                        continue
                    if abs(f1 - f2) <= 1 and abs(r1 - r2) <= 1:
                        s = f1 * size + r1
                        t = f2 * size + r2
                        senders.append(s)
                        receivers.append(t)
                        feats.append([f2 - f1, r2 - r1])
    return np.array(senders), np.array(receivers), np.array(feats, dtype=np.float32)


def _file_edges(size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All directed pairs sharing a file. Features: (delta_rank,)."""
    senders, receivers, feats = [], [], []
    for f in range(size):
        for r1 in range(size):
            for r2 in range(size):
                if r1 == r2:
                    continue
                senders.append(f * size + r1)
                receivers.append(f * size + r2)
                feats.append([r2 - r1])
    return np.array(senders), np.array(receivers), np.array(feats, dtype=np.float32)


def _rank_edges(size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All directed pairs sharing a rank. Features: (delta_file,)."""
    senders, receivers, feats = [], [], []
    for r in range(size):
        for f1 in range(size):
            for f2 in range(size):
                if f1 == f2:
                    continue
                senders.append(f1 * size + r)
                receivers.append(f2 * size + r)
                feats.append([f2 - f1])
    return np.array(senders), np.array(receivers), np.array(feats, dtype=np.float32)


def _diagonal_edges(size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All directed pairs sharing a diagonal (both directions).
    Features: (delta_file, delta_rank)."""
    senders, receivers, feats = [], [], []
    seen = set()
    for f1 in range(size):
        for r1 in range(size):
            for f2 in range(size):
                for r2 in range(size):
                    if (f1, r1) == (f2, r2):
                        continue
                    df, dr = f2 - f1, r2 - r1
                    if abs(df) == abs(dr):
                        pair = (f1 * size + r1, f2 * size + r2)
                        if pair not in seen:
                            seen.add(pair)
                            senders.append(pair[0])
                            receivers.append(pair[1])
                            feats.append([df, dr])
    return np.array(senders), np.array(receivers), np.array(feats, dtype=np.float32)


# Precomputed static edges for both board sizes
_STATIC_EDGES: dict = {}

def _get_static_edges(size: int):
    """Return cached static edges for the given board size."""
    if size not in _STATIC_EDGES:
        _STATIC_EDGES[size] = {
            'grid': _grid_edges(size),
            'file': _file_edges(size),
            'rank': _rank_edges(size),
            'diagonal': _diagonal_edges(size),
        }
    return _STATIC_EDGES[size]


# ---------------------------------------------------------------------------
# Vision edge lookup tables (precomputed per board size)
# ---------------------------------------------------------------------------

def _build_piece_can_reach(size: int) -> np.ndarray:
    """Build boolean table [12, N, N] where piece_can_reach[p, from, to]
    is True if piece type p on square `from` can geometrically reach `to`
    (ignoring blocking for sliding pieces).
    Piece indices: 0-5 = P,N,B,R,Q,K (white), 6-11 = p,n,b,r,q,k (black).
    Square index = file * size + rank (matching _state_nodes rot90 layout)."""
    N = size * size
    table = np.zeros((12, N, N), dtype=bool)
    for sq_from in range(N):
        f1, r1 = sq_from // size, sq_from % size
        for sq_to in range(N):
            if sq_from == sq_to:
                continue
            f2, r2 = sq_to // size, sq_to % size
            df, dr = f2 - f1, r2 - r1
            adf, adr = abs(df), abs(dr)
            # White pawn (index 0): captures diagonally forward (+1 rank)
            if adf == 1 and dr == 1:
                table[0, sq_from, sq_to] = True
            # Black pawn (index 6): captures diagonally backward (-1 rank)
            if adf == 1 and dr == -1:
                table[6, sq_from, sq_to] = True
            # Knight (indices 1, 7)
            if (adf == 1 and adr == 2) or (adf == 2 and adr == 1):
                table[1, sq_from, sq_to] = True
                table[7, sq_from, sq_to] = True
            # Bishop (indices 2, 8)
            if adf == adr and adf > 0:
                table[2, sq_from, sq_to] = True
                table[8, sq_from, sq_to] = True
            # Rook (indices 3, 9)
            if (df == 0 and dr != 0) or (dr == 0 and df != 0):
                table[3, sq_from, sq_to] = True
                table[9, sq_from, sq_to] = True
            # Queen (indices 4, 10)
            if (adf == adr and adf > 0) or (df == 0 and dr != 0) or (dr == 0 and df != 0):
                table[4, sq_from, sq_to] = True
                table[10, sq_from, sq_to] = True
            # King (indices 5, 11)
            if adf <= 1 and adr <= 1:
                table[5, sq_from, sq_to] = True
                table[11, sq_from, sq_to] = True
    return table


def _build_between(size: int) -> np.ndarray:
    """Build table [N, N, max_between] of intermediate squares for sliding pieces.
    between[from, to, :] contains square indices on the ray from->to (exclusive),
    padded with -1. Only meaningful for sliding moves (bishop/rook/queen rays)."""
    N = size * size
    max_between = size - 2  # max intermediates on a ray
    table = np.full((N, N, max_between), -1, dtype=np.int32)
    for sq_from in range(N):
        f1, r1 = sq_from // size, sq_from % size
        for sq_to in range(N):
            if sq_from == sq_to:
                continue
            f2, r2 = sq_to // size, sq_to % size
            df, dr = f2 - f1, r2 - r1
            adf, adr = abs(df), abs(dr)
            # Only for sliding directions
            is_diag = (adf == adr and adf > 0)
            is_straight = (df == 0 and dr != 0) or (dr == 0 and df != 0)
            if not (is_diag or is_straight):
                continue
            step_f = (1 if df > 0 else -1) if df != 0 else 0
            step_r = (1 if dr > 0 else -1) if dr != 0 else 0
            cf, cr = f1 + step_f, r1 + step_r
            idx = 0
            while (cf, cr) != (f2, r2):
                table[sq_from, sq_to, idx] = cf * size + cr
                idx += 1
                cf += step_f
                cr += step_r
    return table


_VISION_TABLES: dict = {}

def _get_vision_tables(size: int):
    if size not in _VISION_TABLES:
        _VISION_TABLES[size] = {
            'can_reach': jnp.array(_build_piece_can_reach(size)),
            'between': jnp.array(_build_between(size)),
        }
    return _VISION_TABLES[size]


# Sliding piece type indices (B, R, Q for both colors)
_SLIDING_TYPES = jnp.array([2, 3, 4, 8, 9, 10])


def _vision_edges(
    observation: jnp.ndarray,
    offset_id: int,
    size: int,
) -> Tuple[EdgeSet, EdgeSet]:
    """Compute attack and defense edges from piece placement.

    Uses observation channels 0-11 (piece placement, one-hot per piece type).
    Returns two EdgeSets (attack, defense) with padded fixed-size arrays.
    Each has features: (delta_file, delta_rank, is_valid, is_my_piece, 12x piece_type).

    Args:
        observation: shape (n_row*n_col, n_features) -- already flattened node features
        offset_id: node offset for batching
        size: board size (5 or 8)
    """
    tables = _get_vision_tables(size)
    can_reach = tables['can_reach']   # (12, N, N)
    between = tables['between']       # (N, N, max_between)

    N = size * size

    # Extract piece placement: channels 0-11 are piece types
    # pieces[sq] = piece_type (0-11) if occupied, -1 if empty
    piece_planes = observation[:, :12]  # (N, 12)
    occupied = piece_planes.sum(axis=-1) > 0.5  # (N,)
    piece_type = jnp.argmax(piece_planes, axis=-1)  # (N,) -- 0-11
    piece_type = jnp.where(occupied, piece_type, -1)

    # White pieces: types 0-5, Black pieces: types 6-11
    is_white = (piece_type >= 0) & (piece_type < 6)
    is_black = (piece_type >= 6) & (piece_type < 12)

    # For all (from, to) pairs, compute vision
    from_sq = jnp.arange(N)[:, None].repeat(N, axis=1).reshape(-1)  # (N*N,)
    to_sq = jnp.arange(N)[None, :].repeat(N, axis=0).reshape(-1)    # (N*N,)

    from_piece = piece_type[from_sq]  # (N*N,)
    from_occupied = occupied[from_sq]
    to_occupied = occupied[to_sq]
    # Can the piece on from_sq geometrically reach to_sq?
    # Gather from can_reach using piece type; invalid piece types map to False
    safe_piece = jnp.clip(from_piece, 0, 11)
    geometric_reach = can_reach[safe_piece, from_sq, to_sq]  # (N*N,)
    geometric_reach = geometric_reach & from_occupied & (from_sq != to_sq)

    # Check blocking for sliding pieces
    is_sliding = jnp.isin(safe_piece, _SLIDING_TYPES)
    between_sqs = between[from_sq, to_sq]  # (N*N, max_between)
    # A square blocks if it's occupied and the between index is valid
    between_occupied = jnp.where(
        between_sqs >= 0,
        occupied[jnp.clip(between_sqs, 0, N - 1)],
        False
    )
    is_blocked = between_occupied.any(axis=-1)  # (N*N,)
    # Only sliding pieces can be blocked
    is_blocked = is_blocked & is_sliding

    can_see = geometric_reach & ~is_blocked  # (N*N,)

    # Classify: attack vs defense
    from_white = is_white[from_sq]
    from_black = is_black[from_sq]
    to_white = is_white[to_sq]
    to_black = is_black[to_sq]
    same_side = (from_white & to_white) | (from_black & to_black)

    is_attack = can_see & (~to_occupied | ~same_side)
    is_defense = can_see & to_occupied & same_side

    # Build features: (df, dr, is_valid, is_my_piece, 12x one-hot piece type)
    f_from = from_sq // size
    r_from = from_sq % size
    f_to = to_sq // size
    r_to = to_sq % size
    delta_file = (f_to - f_from).astype(jnp.float32)
    delta_rank = (r_to - r_from).astype(jnp.float32)
    piece_onehot = jax.nn.one_hot(safe_piece, 12)  # (N*N, 12)

    # For both attack and defense, build features
    # is_my_piece: 1.0 if the piece on `from` belongs to current player (white=channels 0-5)
    is_my_piece = from_white.astype(jnp.float32)

    def _build_edge_set(mask):
        valid = mask.astype(jnp.float32)
        feats = jnp.stack([
            delta_file, delta_rank,
            valid,
            is_my_piece,
        ], axis=-1)  # (N*N, 4)
        feats = jnp.concatenate([feats, piece_onehot], axis=-1)  # (N*N, 16)
        # Zero out features for invalid edges
        feats = feats * valid[:, None]
        senders = jnp.where(mask, from_sq + offset_id, 0).astype(jnp.int32)
        receivers = jnp.where(mask, to_sq + offset_id, 0).astype(jnp.int32)
        return EdgeSet(
            senders=senders,
            receivers=receivers,
            features=feats,
        )

    attack_es = _build_edge_set(is_attack)
    defense_es = _build_edge_set(is_defense)

    return attack_es, defense_es


def _state_nodes(observation: jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    assert(observation.ndim == 3)
    n_row, n_col = observation.shape[:2]
    # cell order is the same as FEN order
    # features = observation.reshape((n_row * n_col, -1))
    # cell order is the same as pgc.Action._from_label
    features = jnp.rot90(observation, k=-1).reshape((n_row * n_col, -1))
    if features.shape[-1] == 115: # Add dummy features for 8x8 compatibility
        zeros = jnp.zeros(features.shape[:-1] + (119,))
        zeros = zeros.at[:,:114].set(features[:,:114])
        features = zeros.at[:,118].set(features[:,114])
    assert(features.shape[-1] == 119)
    return n_row * n_col, features

def _state_edges( # TODO: add self-edge
    legal_action_mask: jnp.ndarray,
    offset_id: int=1
) -> Tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    assert(legal_action_mask.ndim == 1)
    gardner = legal_action_mask.shape[-1] != 4672
    size = 5 if gardner else 8
    _pgc = pgg if gardner else pgc
    # TODO: Missing opponent's moves (pawn moves, castling, ?)
    all_moves = jax.vmap(_pgc.Action._from_label)(
        jnp.arange(legal_action_mask.shape[-1])
    )
    real_moves_id = jnp.where(
        all_moves.to != -1,
        size=455 if gardner else 1858,
        fill_value=-1
    )[0] # The fill_value should never be used
    action_edge_id = jnp.full(legal_action_mask.shape, -1) \
                        .at[real_moves_id] \
                        .set(jnp.arange(real_moves_id.shape[0]))
    all_moves = jax.tree_map(lambda x: x[real_moves_id], all_moves)
    edge_from = all_moves.from_ + offset_id
    edge_to = all_moves.to + offset_id

    n_features = 1+2+4+2*6 # legal, grid offsets, promotions, piece type
    delta_file = (all_moves.to // size) - (all_moves.from_ // size)
    delta_rank = (all_moves.to % size) - (all_moves.from_ % size)
    edge_features = jnp.where(
        (all_moves.to == -1)[:, None].repeat(n_features, axis=-1),
        0,
        jnp.stack([
            # legal
            legal_action_mask[real_moves_id],
            # legal_action_mask,
            # grid offsets
            delta_file,
            delta_rank,
            # promotions
            # doesn't distinguish between pawn promoting to queen and other
            # pieces moving to the 8th rank in a way a pawn could
            (
                  (all_moves.from_ % size == size-2)
                & (all_moves.to % size == size-1)
                & (jnp.abs(all_moves.to // size - all_moves.from_ // size) <= 1)
                & (all_moves.underpromotion == 0)
            ), # queen
            all_moves.underpromotion == 0, # rook
            all_moves.underpromotion == 1, # bishop
            all_moves.underpromotion == 2  # knight
        ] + sum([[ # TODO: add castling
            ( # pawn
                  (jnp.abs(delta_file) <= 1)
                & (
                      (delta_rank == (1 if player else -1))
                    | (
                          (delta_rank == (2 if player else -2))
                        & (all_moves.from_ % size == (1 if player else size-2))
                    ) # TODO: remove this case for gardner chess (no torpedo)
                )
            ),
            ( # knight
                  (jnp.abs(delta_file) == 1) & (jnp.abs(delta_rank) == 2)
                | (jnp.abs(delta_file) == 2) & (jnp.abs(delta_rank) == 1)
            ),
            ( # bishop
                  (jnp.abs(delta_file) == jnp.abs(delta_rank))
                & (all_moves.underpromotion < 0)
            ),
            ( # rook
                  ((jnp.abs(delta_file) == 0) | (jnp.abs(delta_rank) == 0))
                & (all_moves.underpromotion < 0)
            ),
            ( # queen (might be useless as queen == bishop | rook)
                (
                      (jnp.abs(delta_file) == jnp.abs(delta_rank))
                    | ((jnp.abs(delta_file) == 0) | (jnp.abs(delta_rank) == 0))
                )
                & (all_moves.underpromotion < 0)
            ),
            ( # king
                  ((jnp.abs(delta_file) <= 1) & (jnp.abs(delta_rank) <= 1))
                & (all_moves.underpromotion < 0)
            )
        ] for player in range(2)], []), axis=-1)
    )
    return edge_from.shape[0], edge_from, edge_to, edge_features, action_edge_id

def state_to_graph(
    observation: jnp.ndarray,
    legal_action_mask: jnp.ndarray
) -> jraph.GraphsTuple:
    n_nodes, node_features = jax.vmap(_state_nodes)(observation)
    n_nodes = cast(jnp.ndarray, n_nodes)
    node_features = node_features.reshape((-1, node_features.shape[-1]))
    offsets = jnp.concatenate([
        jnp.zeros(1, dtype=jnp.int32),
        n_nodes[:-1]
    ]).cumsum()
    n_edges, moves_from, moves_to, edges_features, action_edge_id = (
        jax.vmap(_state_edges)(
            legal_action_mask,
            offset_id=cast(int, offsets)
        )
    )
    moves_from = moves_from.reshape((-1,)).astype(jnp.int32)
    moves_to = moves_to.reshape((-1,)).astype(jnp.int32)
    edge_offsets = jnp.arange(action_edge_id.shape[0]) * edges_features.shape[1]
    edges_features = edges_features.reshape((-1,) + edges_features.shape[2:])

    edge_offsets = edge_offsets.repeat(action_edge_id.shape[1])
    action_edge_id = action_edge_id.reshape((-1,))
    action_edge_id = jnp.where(
        action_edge_id == -1,
        -1,
        action_edge_id + edge_offsets
    )

    return jraph.GraphsTuple(
        n_node=n_nodes,
        nodes=node_features,
        n_edge=cast(jnp.ndarray, n_edges),
        edges=edges_features,
        senders=moves_from,
        receivers=moves_to,
        globals=action_edge_id
    )

def state_to_hetero_graph(
    observation: jnp.ndarray,
    legal_action_mask: jnp.ndarray
) -> HeteroGraph:
    """Build a HeteroGraph from batched observations and legal action masks.

    Args:
        observation: (batch, rows, cols, channels)
        legal_action_mask: (batch, n_actions)
    Returns:
        HeteroGraph with all 7 edge types.
    """
    batch_size = observation.shape[0]
    gardner = legal_action_mask.shape[-1] != 4672
    size = 5 if gardner else 8

    # --- Nodes (same as state_to_graph) ---
    n_nodes, node_features = jax.vmap(_state_nodes)(observation)
    n_nodes = cast(jnp.ndarray, n_nodes)
    flat_nodes = node_features.reshape((-1, node_features.shape[-1]))
    offsets = jnp.concatenate([
        jnp.zeros(1, dtype=jnp.int32),
        n_nodes[:-1]
    ]).cumsum()

    # --- Move edges (existing _state_edges) ---
    _n_edges, moves_from, moves_to, move_feats, action_edge_id = (
        jax.vmap(_state_edges)(
            legal_action_mask,
            offset_id=cast(int, offsets)
        )
    )
    moves_from = moves_from.reshape((-1,)).astype(jnp.int32)
    moves_to = moves_to.reshape((-1,)).astype(jnp.int32)
    edge_offsets = jnp.arange(action_edge_id.shape[0]) * move_feats.shape[1]
    move_feats = move_feats.reshape((-1,) + move_feats.shape[2:])
    edge_offsets_flat = edge_offsets.repeat(action_edge_id.shape[1])
    action_edge_id = action_edge_id.reshape((-1,))
    action_edge_id = jnp.where(
        action_edge_id == -1,
        -1,
        action_edge_id + edge_offsets_flat
    )
    move_es = EdgeSet(senders=moves_from, receivers=moves_to, features=move_feats)

    # --- Static edges (grid, file, rank, diagonal) ---
    static = _get_static_edges(size)

    def _static_to_edgeset(key):
        s, r, f = static[key]
        s_j, r_j, f_j = jnp.array(s), jnp.array(r), jnp.array(f, dtype=jnp.float32)
        # Tile for each batch element with offset
        all_s = (s_j[None, :] + offsets[:, None]).reshape(-1).astype(jnp.int32)
        all_r = (r_j[None, :] + offsets[:, None]).reshape(-1).astype(jnp.int32)
        all_f = jnp.tile(f_j, (batch_size, 1))
        return EdgeSet(senders=all_s, receivers=all_r, features=all_f)

    grid_es = _static_to_edgeset('grid')
    file_es = _static_to_edgeset('file')
    rank_es = _static_to_edgeset('rank')
    diag_es = _static_to_edgeset('diagonal')

    # --- Vision edges (attack + defense) ---
    def _compute_vision_single(node_feats_single, offset):
        return _vision_edges(node_feats_single, offset, size)

    attack_list, defense_list = jax.vmap(_compute_vision_single)(
        node_features,  # (batch, N, feat_dim)
        offsets          # (batch,)
    )
    # Flatten across batch
    attack_es = EdgeSet(
        senders=attack_list.senders.reshape(-1).astype(jnp.int32),
        receivers=attack_list.receivers.reshape(-1).astype(jnp.int32),
        features=attack_list.features.reshape(-1, attack_list.features.shape[-1]),
    )
    defense_es = EdgeSet(
        senders=defense_list.senders.reshape(-1).astype(jnp.int32),
        receivers=defense_list.receivers.reshape(-1).astype(jnp.int32),
        features=defense_list.features.reshape(-1, defense_list.features.shape[-1]),
    )

    return HeteroGraph(
        n_node=n_nodes,
        nodes=flat_nodes,
        move_edges=move_es,
        grid_edges=grid_es,
        attack_edges=attack_es,
        defense_edges=defense_es,
        file_edges=file_es,
        rank_edges=rank_es,
        diagonal_edges=diag_es,
        globals=action_edge_id,
    )


def main():
    env = pgx.make("gardner_chess")
    state = env.init(jax.random.PRNGKey(0))
    state = jax.tree_map(lambda x: x[None], state)

    x = jax.jit(state_to_graph)(state.observation, state.legal_action_mask)
    # pprint(x)
    # pprint(x.n_node)
    # pprint(x.nodes.shape)
    # pprint(np.array(list(" PNBRQKpnbrqk"))[jnp.rot90((x.nodes[1:,:12] * jnp.arange(1, 13)).sum(axis=-1).reshape((-1,8,8)), axes=(1,2)).astype(jnp.int32)])
    pprint(x.nodes.shape)
    print('   l Δx Δy pq pr pb pn  p  n  b  r  q  k  P  N  B  R  Q  K ')
    print(x.edges[jnp.where(state.legal_action_mask.reshape((-1,)))])
    pprint(jnp.where(state.legal_action_mask.reshape((-1,))))

    # states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    # x = jax.jit(state_to_graph)(states.observation, states.legal_action_mask)
    # pprint((x.n_node, x.n_edge, x.nodes.shape, x.edges.shape))
    # pprint(np.array(list(" PNBRQKpnbrqk"))[jnp.rot90((x.nodes[1:,:12] * jnp.arange(1, 13)).sum(axis=-1).reshape((-1,8,8)), axes=(1,2)).astype(jnp.int32)])
    # print('   l Δx Δy pq pr pb pn  p  n  b  r  q  k  P  N  B  R  Q  K ')
    # print(x.edges[jnp.where(state.legal_action_mask.reshape((-1)))])
    # print(x.receivers[jnp.where(states.legal_action_mask.reshape((-1,)))])
    # pprint((x.receivers.shape, x.receivers.min(), x.receivers.max()))

if __name__ == "__main__":
    main()
