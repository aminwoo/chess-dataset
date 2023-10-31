import chess
import numpy as np
from src.features.policy_index import policy_index
from src.features.board2planes import board2planes, mirrorMove
from src.domain.tcn import tcn_decode


class Game:
    def __init__(self, tcn, white, black, time_control=60):
        self.moves = tcn_decode(tcn)
        self.white = white
        self.black = black
        self.board = chess.Board()
        self.time_control = time_control
        self.move_ptr = 0

    def over(self):
        return self.move_ptr >= len(self.moves)

    def next(self):
        planes = board2planes(self.board)
        move = self.moves[self.move_ptr]
        moves_left = len(self.moves) - self.move_ptr
        self.board.push(move)

        if self.move_ptr % 2 == 0:
            us = self.white
            them = self.black
        else:
            us = self.black
            them = self.white
            move = mirrorMove(move)

        policy = np.zeros(len(policy_index))
        move_uci = move.uci()
        if move_uci[-1] == "n":
            move_uci = move_uci[:-1]

        policy[policy_index.index(move_uci)] = 1

        if us["result"] == "win":
            wdl = (1, 0, 0)
        elif them["result"] == "win":
            wdl = (0, 0, 1)
        else:
            wdl = (0, 1, 0)

        self.move_ptr += 1
        return planes, policy, wdl, moves_left, us
