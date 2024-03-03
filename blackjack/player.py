from blackjack.hand import Hand
from typing import Union
from blackjack.utils import GameState


class Player:
    def __init__(
            self,
            name: Union[str, int] = 0,
            hand: Hand = None,
            stake: int = 1
    ):
        self.name = name
        hand = hand if hand is not None else Hand(stake=stake, player_name=name)
        self.hands = [hand]
        self.decision_hist = []
        self.original_stake = stake

    def reset(self):
        self.hands = [Hand(stake=self.original_stake, player_name=self.name)]

    @property
    def hand(self) -> Hand:
        return self.hands[0]

    @property
    def status(self) -> GameState:
        if any(not hand.is_finished for hand in self.hands) or not len(self.hands):
            return GameState.LIVE

        if self.profit == 0:
            return GameState.DRAW
        elif self.profit > 0:
            return GameState.WON
        elif self.profit < 0:
            return GameState.LOST

    @property
    def profit(self):
        return sum(hand.profit for hand in self.hands)

    @property
    def stake(self):
        return sum(hand.stake for hand in self.hands)

    @property
    def is_finished(self) -> bool:
        return all(hand.is_finished for hand in self.hands)

    def __repr__(self):
        return f'Player {self.name}: hands={self.hands} [{self.status}]'
