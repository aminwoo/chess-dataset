import boto3
import json
import logging
from tcn import tcn_decode


class Game:
    def __init__(self, tcn, white, black, time_control=60):
        self.moves = tcn_decode(tcn)
        self.white = white
        self.black = black
        self.time_control = time_control


if __name__ == '__main__':
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('chess-dataset')
    game_count = 0
    for obj in bucket.objects.all():
        key = obj.key
        body = obj.get()['Body'].read().decode('utf-8')
        games = json.loads(body)['games']
        for game in games:
            game = Game(**game)
            game_count += 1
    print(game_count)

    '''s3object = s3.Object('chess-dataset', 'archives/danielnaroditsky/2010-06.json')
    file_content = s3object.get()['Body'].read().decode('utf-8')
    json_content = json.loads(file_content)
    for game in json_content['games']:
        #print(game.keys())
        #print(game['tcn'])
        game = Game(**game)'''
