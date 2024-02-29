from utils import GameState
from hand import Hand
from card import Card
from typing import Union
from copy import deepcopy


class Player:
    def __init__(
            self,
            name: Union[str, int],
            hand: Hand = None,
            stake: int = 1
    ):
        self.name = name
        self.hand = hand if hand is not None else Hand()
        self.status = GameState.LIVE
        self.decision_hist = []
        self.stake = stake
        self.is_doubled = False

    def deal_card(self, card: Card) -> None:
        self.hand.add_card(card)

    def log_decision(self, decision: str) -> None:
        self.decision_hist.append((deepcopy(self.hand), decision))

    @property
    def card_count(self) -> int:
        return len(self.hand)

    @property
    def is_blackjack(self) -> bool:
        return self.hand.value == 21 and len(self.hand) == 2

    @property
    def is_finished(self) -> bool:
        return self.status in ['WON', 'LOST', 'DRAW']

    @property
    def profit(self) -> float:
        if self.status == GameState.WON and self.is_blackjack:
            return 1.5 * self.stake
        if self.status == GameState.WON:
            return self.stake
        if self.status == GameState.DRAW:
            return 0
        if self.status == GameState.LOST:
            return -self.stake
        return 0

    def __repr__(self):
        return f'Player {self.name}: hand={self.hand} ({self.hand.value}) [{self.status}]'
