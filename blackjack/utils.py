import os
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import json

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data')


def card2rank(c):
    if 2 <= c <= 9:
        return str(c)
    if c == 1 or c == 11:
        return 'A'
    if c == 10:
        return 'T'


@dataclass
class GameDecision(str, Enum):
    SPLIT: str = 'split'
    STAND: str = 'stand'
    HIT: str = 'hit'
    DOUBLE: str = 'double'


@dataclass
class GameState(str, Enum):
    WON: str = 'WON'
    LOST: str = 'LOST'
    DRAW: str = 'DRAW'
    LIVE: str = 'LIVE'
    STAND: str = 'STAND'


def csv_to_dict(fn: str) -> dict:
    df_bs = pd.read_csv(fn, index_col=0)
    print(df_bs)
    json_bs = json.loads(df_bs.to_json())

    bs = {
        (int(player) if player.isnumeric() else player, int(dealer)): decision
        for dealer, player_decision in json_bs.items()
        for player, decision in player_decision.items()
    }
    return bs


count_2combinations = {
    i: ['{}{}'.format(card2rank(j), card2rank(i-j))
        for j in range(2, min((i-1)//2+1, 10)) if i-j != j and i-j < 11]
    for i in range(5, 20)
}
count_2combinations.update({20: '884', 21: '993', 2: '2', 3: '3', 4: '4'})
