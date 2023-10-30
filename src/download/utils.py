import requests
import boto3
import json
import logging
import threading
from botocore.exceptions import ClientError
from tqdm import tqdm


def get_players(path):
    players = []
    with open(path) as f:
        for player in f.readlines():
            players.append(player.strip())

    return players


def get_archive_urls(players):
    urls = {}
    for player in players:
        r = requests.get(f'https://api.chess.com/pub/player/{player}/games/archives',
                         headers={
                             'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                           'Chrome/109.0.0.0'
                         }, )
        urls[player] = r.json()['archives']

    return urls


def upload_to_s3(path, obj):
    s3 = boto3.resource('s3')
    try:
        s3object = s3.Object('chess-dataset', path)
        s3object.put(Body=(bytes(json.dumps(obj).encode('UTF-8'))))
    except ClientError as e:
        logging.error(e)
        return False
    return True


def download_archives(urls):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('chess-dataset')
    objs = list(bucket.objects.filter(Prefix='archives'))
    existing_paths = set(i.key for i in objs)

    for player in urls:
        for url in tqdm(urls[player]):
            s3path = f"archives/{player.lower()}/{'-'.join(url.split('/')[-2:])}.json"
            if s3path in existing_paths:
                continue

            r = requests.get(url,
                             headers={
                                 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                               'Chrome/109.0.0.0'
                             }, )

            t = threading.Thread(target=upload_to_s3, args=(s3path, r.json(),))
            t.start()


if __name__ == '__main__':
    players = get_players('../../data/players.txt')
    urls = get_archive_urls(players)
    download_archives(urls)
