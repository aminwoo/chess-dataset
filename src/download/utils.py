"""Utility functions to download from chess.com and store to s3 bucket."""
import json
import logging
import threading
import requests
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/109.0.0.0"
}


def get_users(path):
    """
    Get all users on file.

    :param path:
    :return:
    """
    users = []
    with open(path, encoding="utf-8") as f:
        for user in f.readlines():
            users.append(user.strip())

    return users


def get_archive_urls(users):
    """

    :param users:
    :return:
    """
    urls = {}
    for user in users:
        r = requests.get(
            f"https://api.chess.com/pub/player/{user}/games/archives", headers=headers, timeout=10
        )
        urls[user] = r.json()["archives"]
    return urls


def upload_to_s3(path, obj):
    """
    Upload object to s3 path.

    :return: Success
    """
    s3resource = boto3.resource("s3")
    try:
        s3object = s3resource.Object("chess-dataset", path)
        s3object.put(Body=(bytes(json.dumps(obj).encode("UTF-8"))))
    except ClientError as e:
        logging.error(e)
        return False
    return True


def download_archives(urls):
    """

    :param urls:
    :return:
    """
    s3resource = boto3.resource("s3")
    bucket = s3resource.Bucket("chess-dataset")
    objs = list(bucket.objects.filter(Prefix="archives"))
    existing_paths = set(i.key for i in objs)

    for player in urls:
        for url in tqdm(urls[player]):
            s3path = f"archives/{player.lower()}/{'-'.join(url.split('/')[-2:])}.json"
            if s3path in existing_paths:
                continue
            r = requests.get(url, headers=headers, timeout=10)

            # Spawn thread to prevent blocking
            t = threading.Thread(
                target=upload_to_s3,
                args=(
                    s3path,
                    r.json(),
                ),
            )
            t.start()


def get_leaderboard(mode="live_blitz", k=50):
    """
    Return top k players fromm leaderboard.

    :param mode:
    :param k:
    :return:
    """
    r = requests.get("https://api.chess.com/pub/leaderboards", headers=headers, timeout=10)
    return [user["username"] for user in r.json()[mode][:k]]


if __name__ == "__main__":
    path = "../../data/players.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines("\n".join(get_leaderboard()))
    download_archives(get_archive_urls(get_users(path)))
