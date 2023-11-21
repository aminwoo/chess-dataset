import logging
import numpy as np
import chess

from src.domain.game import Game, EngineConfig


class Loader:
    """Class to load games in batches."""

    def __init__(self, data, engine_config: EngineConfig = None):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_config.path) if engine_config else None
        # self.engine.configure({"Hash": 1})
        self.games = None
        self.game_ptr = 0
        self.load(data)

    def load(self, data):
        """
        Load in new batch of games.

        :param data:
        :return:
        """
        self.games = [
            Game(obj["tcn"], obj["white"], obj["black"], self.engine) for obj in data if "tcn" in obj
        ]
        self.game_ptr = 0

    def get(self, batch_size=1024):
        """
        Get a batch for neural network training.

        :param batch_size:
        :return:
        """
        planes = np.zeros((batch_size, 112, 8, 8))
        policy = np.zeros((batch_size, 1858))
        wdl = np.zeros((batch_size, 3))
        moves_left = np.zeros((batch_size, 1))

        data_ptr = 0
        while data_ptr < batch_size:
            # Get next unfinished game
            while self.game_ptr < len(self.games) and self.games[self.game_ptr].over():
                self.game_ptr += 1

            # We don't have enough games to fill the next batch
            if self.game_ptr >= len(self.games):
                return False

            try:
                (
                    planes[data_ptr],
                    policy[data_ptr],
                    wdl[data_ptr],
                    moves_left[data_ptr]
                ) = self.games[self.game_ptr].next()
            except chess.engine.EngineError as e:
                logging.error(e)
                self.game_ptr += 1  # Skip to next game
            else:
                data_ptr += 1

        idx = np.random.permutation(batch_size)
        return planes[idx], policy[idx], wdl[idx], moves_left[idx]
