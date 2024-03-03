from blackjack.utils import GameState, GameDecision
from blackjack.card import Card
from typing import Union, List


class Hand:
    def __init__(
            self,
            cards: Union[List[Card], str] = None,
            is_split: bool = False,
            stake: int = 1,
            player_name: str | int = None,
    ):

        self.player_name = player_name
        if cards is None:
            cards = []

        self.cards = [Card(c) if isinstance(c, str) else c for c in cards]
        self.is_split = is_split
        self.is_doubled = False
        self.stake = stake
        self.decision_hist = []

        self.status = GameState.LIVE
        self.round = 0

    def deal_card(self, card: Card) -> None:
        if isinstance(card, str):
            card = Card(card)
        self.cards.append(card)

    def log_decision(self, decision: GameDecision) -> None:
        self.decision_hist.append((str(self), decision.value))

    def value(self, soft=False) -> Union[int, tuple[int, int]]:
        if self.num_aces == 0:
            return self.hard_value

        min_val = self.hard_value + self.num_aces
        max_val = self.hard_value + self.num_aces - 1 + 11
        if 21 in [min_val, max_val]:
            return 21
        if min_val > 21:
            return min_val
        if max_val > 21:
            return min_val

        return (min_val, max_val) if soft else max_val

    @property
    def num_aces(self) -> int:
        return [c for c in self.cards].count('A')

    @property
    def hard_cards(self) -> List[Card]:
        return [c for c in self.cards if c.rank != 'A']

    @property
    def hard_value(self) -> int:
        return sum(c.value for c in self.cards if c != 'A')

    @property
    def is_soft_value(self) -> bool:
        return isinstance(self.value(soft=True), tuple)

    @property
    def is_blackjack(self) -> bool:
        return len(self) == 2 and self.value() == 21 and not self.is_split

    @property
    def is_splittable(self) -> bool:
        return len(self) == 2 and self.cards[0].value == self.cards[1].value

    @property
    def pretty_val(self):
        return ''.join([c.rank for c in self.cards])

    def get_string_rep(self):
        if self.is_soft_value:
            return 'A{}'.format(self.hard_value + self.num_aces - 1)
        return str(self)

    @property
    def is_finished(self):
        return self.status.value in [GameState.WON, GameState.DRAW, GameState.LOST]

    @property
    def is_idle(self):
        return self.is_finished or self.status.value == GameState.STAND

    @property
    def profit(self) -> float:
        if self.status.value == GameState.WON and self.is_blackjack:
            return 1.5 * self.stake
        if self.status.value == GameState.WON:
            return self.stake
        if self.status.value == GameState.DRAW:
            return 0
        if self.status.value == GameState.LOST:
            return -self.stake
        return 0

    def __repr__(self):
        return ''.join([c.rank for c in self.cards]).replace('J', 'T').replace('Q', 'T').replace('K', 'T')

    def __len__(self):
        return len(self.cards)

    def __eq__(self, other):
        return self.is_blackjack and other.is_blackjack or \
            not self.is_blackjack and not other.is_blackjack and self.value() == other.value()

    def __gt__(self, other):
        return self.value() > other.value() or self.is_blackjack and not other.is_blackjack
