import json
import glob
import matplotlib.pyplot as plt
import numpy as np

ratings = []

games = []
files = glob.glob("../../data/games/*")
for file in files:
    with open(file) as f:
        obj = json.load(f)

    for game in obj:
        ratings.append(game["white"]["rating"])
        ratings.append(game["black"]["rating"])
        games.append(game)

print(len(games))


plt.hist(ratings, 100)
plt.show()
print(np.mean(ratings))