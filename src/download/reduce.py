from collections import defaultdict
import boto3
import json
from tqdm import tqdm

remove_columns = ["url", "pgn", "end_time", "rated", "uuid", "time_class", "initial_setup", "fen", "rules"]


def filter(x):
    ret = []
    for i in x:
        if i["rules"] != "chess" or i["initial_setup"] != "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1":
            continue

        time = 0
        if "/" in i["time_control"]:
            time = int(i["time_control"].split("/")[-1])
        elif "+" in i["time_control"]:
            time = int(i["time_control"].split("+")[0])
        else:
            time = int(i["time_control"])
        if time < 180:
            continue

        for j in remove_columns:
            if j in i:
                i.pop(j)
        ret.append(i)
    return ret


if __name__ == "__main__":
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("chess-dataset")
    game_count = 0

    games = defaultdict(lambda: [])

    for obj in tqdm(bucket.objects.all()):
        key = obj.key
        user = key.split("/")[1]
        body = obj.get()["Body"].read().decode("utf-8")
        body = json.loads(body)
        games[user].extend(filter(body["games"]))

    for user in games:
        game_count += len(games[user])
        with open(f"../../data/games/{user}.json", "w") as f:
            json.dump(games[user], f)

    print(game_count)
