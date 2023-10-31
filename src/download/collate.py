"""Retrieve games from s3 and collate games for each user."""
import json
from collections import defaultdict
from tqdm import tqdm
import boto3

remove_columns = [
    "url",
    "pgn",
    "end_time",
    "rated",
    "uuid",
    "time_class",
    "initial_setup",
    "fen",
    "rules",
]


def filter_games(x):
    """

    :param x:
    :return:
    """
    ret = []
    for i in x:
        if (
            i["rules"] != "chess"
            or i["initial_setup"]
            != "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        ):
            continue

        if "/" in i["time_control"]:  # Ignore daily games
            continue
        if (
            int(i["time_control"].split("+")[0]) < 180
        ):  # Ignore games less than 3 minutes
            continue

        for j in remove_columns:
            if j in i:
                i.pop(j)
        ret.append(i)
    return ret


if __name__ == "__main__":
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("chess-dataset")
    games = defaultdict(lambda: [])

    for obj in tqdm(bucket.objects.all()):
        key = obj.key
        user = key.split("/")[1]
        body = obj.get()["Body"].read().decode("utf-8")
        body = json.loads(body)
        try:
            games[user].extend(filter_games(body["games"]))
        except KeyError as e:
            print(body)

    game_count = 0
    for user in games:
        game_count += len(games[user])
        with open(f"../../data/games/{user}.json", "w", encoding="utf-8") as f:
            json.dump(games[user], f)
    print("Total number of games:", game_count)
