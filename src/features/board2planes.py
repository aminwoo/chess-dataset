import chess
import numpy as np
from time import time
from math import exp
from src.features.policy_index import policy_index
import re

MOVE_MAP = dict(list(zip(policy_index, range(len(policy_index)))))

W_PAWN = chess.Piece(chess.PAWN, chess.WHITE)
W_KNIGHT = chess.Piece(chess.KNIGHT, chess.WHITE)
W_BISHOP = chess.Piece(chess.BISHOP, chess.WHITE)
W_ROOK = chess.Piece(chess.ROOK, chess.WHITE)
W_QUEEN = chess.Piece(chess.QUEEN, chess.WHITE)
W_KING = chess.Piece(chess.KING, chess.WHITE)
B_PAWN = chess.Piece(chess.PAWN, chess.BLACK)
B_KNIGHT = chess.Piece(chess.KNIGHT, chess.BLACK)
B_BISHOP = chess.Piece(chess.BISHOP, chess.BLACK)
B_ROOK = chess.Piece(chess.ROOK, chess.BLACK)
B_QUEEN = chess.Piece(chess.QUEEN, chess.BLACK)
B_KING = chess.Piece(chess.KING, chess.BLACK)


def assign_piece(planes, piece_step, row, col):
    planes[piece_step][row][col] = 1


DISPATCH = {str(W_PAWN): lambda retval, row, col: assign_piece(retval, 0, row, col),
            str(W_KNIGHT): lambda retval, row, col: assign_piece(retval, 1, row, col),
            str(W_BISHOP): lambda retval, row, col: assign_piece(retval, 2, row, col),
            str(W_ROOK): lambda retval, row, col: assign_piece(retval, 3, row, col),
            str(W_QUEEN): lambda retval, row, col: assign_piece(retval, 4, row, col),
            str(W_KING): lambda retval, row, col: assign_piece(retval, 5, row, col),
            str(B_PAWN): lambda retval, row, col: assign_piece(retval, 6, row, col),
            str(B_KNIGHT): lambda retval, row, col: assign_piece(retval, 7, row, col),
            str(B_BISHOP): lambda retval, row, col: assign_piece(retval, 8, row, col),
            str(B_ROOK): lambda retval, row, col: assign_piece(retval, 9, row, col),
            str(B_QUEEN): lambda retval, row, col: assign_piece(retval, 10, row, col),
            str(B_KING): lambda retval, row, col: assign_piece(retval, 11, row, col)}

MOVE_RE = re.compile(r"^([a-h])(\d)([a-h])(\d)(.*)$")


def mirrorMoveUCI(move):
    m = MOVE_RE.match(move)
    return "{}{}{}{}{}".format(m.group(1), 9 - int(m.group(2)), m.group(3), 9 - int(m.group(4)), m.group(5))


def mirrorMove(move):
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)


def append_plane(planes, ones):
    if ones:
        return np.append(planes, np.ones((1, 8, 8), dtype=float), axis=0)
    else:
        return np.append(planes, np.zeros((1, 8, 8), dtype=float), axis=0)


def board2planes(board_):
    if not board_.turn:
        board = board_.mirror()
    else:
        board = board_

    retval = np.zeros((13, 8, 8), dtype=float)
    for row in range(8):
        for col in range(8):
            piece = str(board.piece_at(chess.SQUARES[row * 8 + col]))
            if piece != "None":
                DISPATCH[piece](retval, row, col)

    temp = np.copy(retval)
    for i in range(7):
        retval = np.append(retval, temp, axis=0)

    retval = append_plane(retval, bool(board.castling_rights & chess.BB_H1))
    retval = append_plane(retval, bool(board.castling_rights & chess.BB_A1))
    retval = append_plane(retval, bool(board.castling_rights & chess.BB_H8))
    retval = append_plane(retval, bool(board.castling_rights & chess.BB_A8))
    retval = append_plane(retval, not board_.turn)

    retval = append_plane(retval, False)
    retval = append_plane(retval, False)
    retval = append_plane(retval, True)
    return np.expand_dims(retval, axis=0)


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
    board = chess.Board(fen="rnbqkb1r/ppp1pppp/5n2/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")

    start = time()
    REPS = 10000

    for i in range(0, REPS):
        planes = board2planes(board)

    end = time()

    print(end - start)
    print((end - start) / REPS)
    print(planes.shape)
