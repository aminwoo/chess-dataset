from dataclasses import dataclass
import numpy as np
import chess
import chess.engine

from src.features.policy_index import policy_index
from src.features.board2planes import board2planes, mirror_move
from src.domain.tcn import tcn_decode


@dataclass
class EngineConfig:
    """Class to specify other engine configuration."""
    path: str = "engines/stockfish-windows-x86-64-avx2.exe"
    time: float = 0.1
    nodes: int = 100000


class Game:
    """This class represents one chess game and provides methods to load the positions."""
    def __init__(self, tcn_moves, white_info, black_info, engine):
        self.moves = tcn_decode(tcn_moves)
        self.white_info = white_info
        self.black_info = black_info
        self.engine = engine
        self.board = chess.Board()
        self.move_ptr = 0

    def over(self) -> bool:
        """
        Returns true if we are at the end of the game.

        :return:
        """
        return self.move_ptr >= len(self.moves)

    def next(self):
        """Get the current board in plane representation and policy planes for next move."""
        planes = board2planes(self.board)
        move = self.moves[self.move_ptr]
        moves_left = len(self.moves) - self.move_ptr

        wins = 0
        draws = 0
        losses = 0

        if self.engine:
            result = self.engine.play(self.board, chess.engine.Limit(nodes=10000), info=chess.engine.INFO_SCORE)
            self.board.push(move)
            move = result.move
            wdl = result.info["score"].white().wdl()
            wins, draws, losses = wdl
            wins /= wdl.total()
            draws /= wdl.total()
            losses /= wdl.total()
        else:
            self.board.push(move)
            if self.white_info["result"] == "win":
                wins = 1
            elif self.black_info["result"] == "win":
                losses = 1
            else:
                draws = 0

        if self.move_ptr % 2:
            move = mirror_move(move)
            wins, losses = losses, wins

        self.move_ptr += 1

        policy = np.zeros(len(policy_index))
        move_uci = move.uci()
        # For knight promotions don't specify promotion type
        if move_uci[-1] == "n":
            move_uci = move_uci[:-1]
        policy[policy_index.index(move_uci)] = 1

        return planes, policy, (wins, draws, losses), moves_left
