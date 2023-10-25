import json
import numpy as np
from src.domain.game import Game


class Loader:
    def __init__(self, data):
        self.games = [Game(obj["tcn"], obj["white"], obj["black"]) for obj in data if obj["rules"] == "chess"]
        self.game_ptr = 0

    def get(self, batch_size=1024):
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

            planes[data_ptr], policy[data_ptr], wdl[data_ptr], moves_left[data_ptr] = self.games[self.game_ptr].next()
            data_ptr += 1

        return planes, policy, wdl, moves_left


if __name__ == "__main__":
    with open("../../data/games/2023-09.json") as f:
        data = json.load(f)

    loader = Loader(data["games"])
    while loader.get():
        pass