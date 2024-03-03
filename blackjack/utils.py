import os
from dataclasses import dataclass
from enum import Enum
from blackjack.hand import Hand

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data')


def format_hand(hand: Hand):
    if hand.is_splittable:
        player_hand = str(hand)

        if player_hand.isnumeric():
            player_hand = int(player_hand)

        if player_hand == 'AA':
            player_hand = 12

    elif hand.is_soft_value:
        player_hand = hand.get_string_rep()
    else:
        player_hand = hand.value()

    return player_hand

def card2rank(c):
    if 2 <= c <= 9:
        return str(c)
    if c == 1 or c == 11:
        return 'A'
    if c == 10:
        return 'T'


count_2combinations = {
    i: ['{}{}'.format(card2rank(j), card2rank(i-j))
        for j in range(2, min((i-1)//2+1, 10)) if i-j != j and i-j < 11]
    for i in range(5, 20)
}


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
