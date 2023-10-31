import chess
import numpy as np
from time import time
from math import exp
from src.features.policy_index import policy_index
import re

MOVE_MAP = dict(list(zip(policy_index, range(len(policy_index)))))
MOVE_RE = re.compile(r"^([a-h])(\d)([a-h])(\d)(.*)$")


def mirrorMoveUCI(move):
    m = MOVE_RE.match(move)
    return "{}{}{}{}{}".format(
        m.group(1), 9 - int(m.group(2)), m.group(3), 9 - int(m.group(4)), m.group(5)
    )


def mirrorMove(move):
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        move.promotion,
    )


def append_plane(planes, ones):
    if ones:
        return np.append(planes, np.ones((1, 64), dtype=float), axis=0)
    else:
        return np.append(planes, np.zeros((1, 64), dtype=float), axis=0)


def board2planes(board_):
    board = board_.copy()
    repetitions = []
    for i in range(8):
        repetitions.append(board.is_repetition(count=2))
        if board.move_stack:
            board.pop()

    if not board_.turn:
        board = board_.mirror().copy()
    else:
        board = board_.copy()

    planes = np.zeros((104, 64), dtype=float)
    for i in range(8):
        base = i * 13
        planes[base + 0][list(board.pieces(chess.PAWN, board.turn))] = 1
        planes[base + 1][list(board.pieces(chess.KNIGHT, board.turn))] = 1
        planes[base + 2][list(board.pieces(chess.BISHOP, board.turn))] = 1
        planes[base + 3][list(board.pieces(chess.ROOK, board.turn))] = 1
        planes[base + 4][list(board.pieces(chess.QUEEN, board.turn))] = 1
        planes[base + 5][list(board.pieces(chess.KING, board.turn))] = 1

        planes[base + 6][list(board.pieces(chess.PAWN, not board.turn))] = 1
        planes[base + 7][list(board.pieces(chess.KNIGHT, not board.turn))] = 1
        planes[base + 8][list(board.pieces(chess.BISHOP, not board.turn))] = 1
        planes[base + 9][list(board.pieces(chess.ROOK, not board.turn))] = 1
        planes[base + 10][list(board.pieces(chess.QUEEN, not board.turn))] = 1
        planes[base + 11][list(board.pieces(chess.KING, not board.turn))] = 1

        if repetitions[i]:
            planes[base + 12] = 1

        if board.move_stack:
            board.pop()

    planes = append_plane(planes, bool(board.castling_rights & chess.BB_A1))
    planes = append_plane(planes, bool(board.castling_rights & chess.BB_H1))
    planes = append_plane(planes, bool(board.castling_rights & chess.BB_A8))
    planes = append_plane(planes, bool(board.castling_rights & chess.BB_H8))
    planes = append_plane(planes, not board_.turn)

    planes = append_plane(planes, board.ply())
    planes = append_plane(planes, False)
    planes = append_plane(planes, True)
    return np.expand_dims(planes.reshape(112, 8, 8), axis=0)


def policy2moves(board_, policy_tensor, softmax_temp=1.61):
    if not board_.turn:
        board = board_.mirror()
    else:
        board = board_
    policy = policy_tensor.numpy()

    moves = list(board.legal_moves)
    retval = {}
    max_p = float("-inf")
    for m in moves:
        uci = m.uci()
        fixed_uci = uci
        if uci == "e1g1" and board.is_kingside_castling(m):
            fixed_uci = "e1h1"
        elif uci == "e1c1" and board.is_queenside_castling(m):
            fixed_uci = "e1a1"
        if uci[-1] == "n":
            # we are promoting to knight, so trim the character
            fixed_uci = uci[0:-1]
        # now mirror the uci
        if not board_.turn:
            uci = mirrorMoveUCI(uci)
        p = policy[0][MOVE_MAP[fixed_uci]]
        retval[uci] = p
        if p > max_p:
            max_p = p
    total = 0.0
    for uci in retval:
        retval[uci] = exp((retval[uci] - max_p) / softmax_temp)
        total = total + retval[uci]

    if total > 0.0:
        for uci in retval:
            retval[uci] = retval[uci] / total
    return retval


if __name__ == "__main__":
    board = chess.Board(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    board.push(chess.Move.from_uci("g1f3"))
    board.push(chess.Move.from_uci("g8f6"))
    board.push(chess.Move.from_uci("f3g1"))
    board.push(chess.Move.from_uci("f6g8"))
    board.push(chess.Move.from_uci("g1f3"))
    #board.push(chess.Move.from_uci("g8f6"))
    #board.push(chess.Move.from_uci("f3g1"))
    #board.push(chess.Move.from_uci("f6g8"))

    start = time()
    REPS = 10000

    for i in range(0, REPS):
        planes = board2planes(board)

    end = time()

    print(end - start)
    print((end - start) / REPS)
    print(planes[0][12])
