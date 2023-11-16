import numpy as np
from src.domain.game import Game, EngineConfig


class Loader:
    """Class to load games in batches."""
    def __init__(self, data):
        self.games = [
            Game(obj["tcn"], obj["white"], obj["black"], EngineConfig) for obj in data if "tcn" in obj
        ]
        self.game_ptr = 0

    def get(self, batch_size=1024):
        """

        :param batch_size:
        :return:
        """
        planes = np.zeros((batch_size, 112, 8, 8))
        policy = np.zeros((batch_size, 1858))
        wdl = np.zeros((batch_size, 3))
        moves_left = np.zeros((batch_size, 1))

        data_ptr = 0
        while data_ptr < batch_size:
            while self.game_ptr < len(self.games) and self.games[self.game_ptr].over():
                self.game_ptr += 1

            if self.game_ptr >= len(self.games):
                return False

            (
                planes[data_ptr],
                policy[data_ptr],
                wdl[data_ptr],
                moves_left[data_ptr]
            ) = self.games[self.game_ptr].next()
            data_ptr += 1

        idx = np.random.permutation(batch_size)
        return planes[idx], policy[idx], wdl[idx], moves_left[idx]
