"""Bidirectional conversion between pgx action indices and python-chess Move objects.

pgx square encoding (column-major): square = file * 8 + rank
    A1=0, A2=1, ..., A8=7, B1=8, ...
python-chess square encoding (row-major): square = rank * 8 + file
    A1=0, B1=1, ..., H1=7, A2=8, ...

pgx flips the board for black (rank-only flip): flipped = (sq // 8) * 8 + (7 - sq % 8)

Action encoding (from utils.py):
    action = from_square * 73 + plane_index
    Planes 0-8:  underpromotions (3 piece types x 3 directions)
                 piece = plane // 3  (0=Rook, 1=Bishop, 2=Knight)
    Planes 9-72: queen moves (8 dirs x 7 distances) + knight moves (8)
    TO_MAP[from_sq, plane] gives destination square
    PLANE_MAP[from_sq, to_sq] gives plane for queen/knight moves
"""

from typing import Optional

import chess
import numpy as np
import pgx.chess as pgc


# Precomputed lookup: pgx_action -> (from_sq_pgx, to_sq_pgx, underpromotion_index)
# underpromotion_index: -1=none/queen, 0=rook, 1=bishop, 2=knight
_ACTION_TABLE: Optional[np.ndarray] = None

# Reverse lookup: (from_sq_pgx, to_sq_pgx) -> list of (plane, underpromotion_index)
_REVERSE_TABLE: Optional[dict] = None


def _build_tables():
    global _ACTION_TABLE, _REVERSE_TABLE

    to_map = np.array(pgc.TO_MAP)  # (64, 73)
    n_actions = 64 * 73  # 4672

    # action_table[action] = (from_sq, to_sq, underpromotion)
    table = np.zeros((n_actions, 3), dtype=np.int32)
    reverse = {}

    for from_sq in range(64):
        for plane in range(73):
            action = from_sq * 73 + plane
            to_sq = int(to_map[from_sq, plane])
            if plane < 9:
                underpromo = plane // 3  # 0=R, 1=B, 2=N
            else:
                underpromo = -1
            table[action] = [from_sq, to_sq, underpromo]

            if to_sq >= 0:
                key = (from_sq, to_sq)
                if key not in reverse:
                    reverse[key] = []
                reverse[key].append((plane, underpromo))

    _ACTION_TABLE = table
    _REVERSE_TABLE = reverse


def _ensure_tables():
    if _ACTION_TABLE is None:
        _build_tables()


def _pgx_flip_sq(sq: int) -> int:
    """Flip a pgx square (rank-only flip for black perspective)."""
    return (sq // 8) * 8 + (7 - sq % 8)


def _pgx_to_chess_sq(pgx_sq: int) -> int:
    """Convert pgx square (file*8+rank) to python-chess square (rank*8+file)."""
    file = pgx_sq // 8
    rank = pgx_sq % 8
    return rank * 8 + file


def _chess_to_pgx_sq(chess_sq: int) -> int:
    """Convert python-chess square (rank*8+file) to pgx square (file*8+rank)."""
    file = chess_sq % 8
    rank = chess_sq // 8
    return file * 8 + rank


# Underpromotion index to python-chess piece type
_UNDERPRO_TO_PIECE = {
    0: chess.ROOK,
    1: chess.BISHOP,
    2: chess.KNIGHT,
}

# python-chess piece type to underpromotion index
_PIECE_TO_UNDERPRO = {v: k for k, v in _UNDERPRO_TO_PIECE.items()}


def pgx_action_to_chess_move(
    pgx_action: int,
    board: chess.Board,
    is_black_turn: bool,
) -> chess.Move:
    """Convert a pgx action index to a python-chess Move.

    Args:
        pgx_action: The pgx action index (0..4671).
        board: Current python-chess Board (for determining promotion context).
        is_black_turn: Whether it's black's turn (pgx flips the board for black).

    Returns:
        A python-chess Move object.
    """
    _ensure_tables()
    assert _ACTION_TABLE is not None

    from_sq_pgx = int(_ACTION_TABLE[pgx_action, 0])
    to_sq_pgx = int(_ACTION_TABLE[pgx_action, 1])
    underpromo = int(_ACTION_TABLE[pgx_action, 2])

    # Un-flip for black
    if is_black_turn:
        from_sq_pgx = _pgx_flip_sq(from_sq_pgx)
        to_sq_pgx = _pgx_flip_sq(to_sq_pgx)

    # Convert to python-chess squares
    from_sq_chess = _pgx_to_chess_sq(from_sq_pgx)
    to_sq_chess = _pgx_to_chess_sq(to_sq_pgx)

    # Determine promotion
    promotion = None
    piece = board.piece_at(from_sq_chess)
    if piece is not None and piece.piece_type == chess.PAWN:
        to_rank = chess.square_rank(to_sq_chess)
        if to_rank == 7 or to_rank == 0:  # reaching promotion rank
            if underpromo >= 0:
                promotion = _UNDERPRO_TO_PIECE[underpromo]
            else:
                promotion = chess.QUEEN

    return chess.Move(from_sq_chess, to_sq_chess, promotion=promotion)


def chess_move_to_pgx_action(
    move: chess.Move,
    board: chess.Board,
    is_black_turn: bool,
) -> int:
    """Convert a python-chess Move to a pgx action index.

    Args:
        move: A python-chess Move.
        board: Current python-chess Board.
        is_black_turn: Whether it's black's turn.

    Returns:
        The pgx action index (0..4671).
    """
    _ensure_tables()
    assert _REVERSE_TABLE is not None

    from_sq_chess = move.from_square
    to_sq_chess = move.to_square

    # Convert to pgx squares
    from_sq_pgx = _chess_to_pgx_sq(from_sq_chess)
    to_sq_pgx = _chess_to_pgx_sq(to_sq_chess)

    # Flip for black
    if is_black_turn:
        from_sq_pgx = _pgx_flip_sq(from_sq_pgx)
        to_sq_pgx = _pgx_flip_sq(to_sq_pgx)

    key = (from_sq_pgx, to_sq_pgx)
    candidates = _REVERSE_TABLE.get(key, [])

    if not candidates:
        raise ValueError(
            f"No pgx action found for move {move} "
            f"(pgx from={from_sq_pgx}, to={to_sq_pgx})"
        )

    # Determine which plane to use based on promotion
    if move.promotion is not None and move.promotion != chess.QUEEN:
        target_underpromo = _PIECE_TO_UNDERPRO.get(move.promotion, -1)
    else:
        target_underpromo = -1  # queen promotion or no promotion

    for plane, underpromo in candidates:
        if underpromo == target_underpromo:
            return from_sq_pgx * 73 + plane

    # Fallback: if no exact underpromotion match, use first non-underpromotion
    for plane, underpromo in candidates:
        if underpromo == -1:
            return from_sq_pgx * 73 + plane

    raise ValueError(
        f"No matching pgx action for move {move} with promotion={move.promotion}"
    )
