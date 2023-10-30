import requests

r = requests.get('https://api.chess.com/pub/leaderboards',
                 headers={
                     'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                   'Chrome/109.0.0.0'
                 }, )

users = [i['username'] for i in r.json()['live_blitz']]
with open('../../data/players.txt', 'w') as f:
    f.writelines('\n'.join(users))