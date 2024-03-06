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
        """

        :param cards: Cards the Hand contains
        :param is_split: Bool indicating whether this hand resulted from a split hand
        :param stake: Original amount wagered on this hand
        :param player_name: optional name of the player the hand is owned by.
        """

        self.player_name = player_name
        if cards is None:
            cards = []

        self.cards = [Card(c) if isinstance(c, str) else c for c in cards]
        self.is_split = is_split
        self.is_doubled = False
        self.original_stake = stake
        self.stake = stake
        self.decision_hist = []

        self.status = GameState.LIVE
        self.round = 0

    def deal_card(self, card: Card) -> None:
        """
        :param card: Card object to be added to the list of Cards of the Hand
        :return: None
        """
        if isinstance(card, str):
            card = Card(card)
        self.cards.append(card)

    def log_decision(self, decision: GameDecision) -> None:
        """
        Appends the decision and current game state to an array for a historical tracking of the decisions taken.
        :param decision: GameDecision object with the decision
        :return:
        """
        self.decision_hist.append((str(self), decision.value))

    def value(self, soft=False) -> Union[int, tuple[int, int]]:
        """
        For non-Aces, it is a matter of counting the value of each card and taking the sum.

        If an Ace is part of the Hand, we count it both as 1 and 11. If both possibilities are less than 21,
        then both are valid values, and both values are returned as a tuple if soft=True.

        If one of the soft values is 21, it does not make sense to consider the other value, whatever it is.

        At most one Ace can count as 11, otherwise the value would exceed 21. Hence, it is automatically assumed that the second Ace onwards has a hard value of 1.
        :param soft: bool indicating the best value of the hand in the case of a soft hand
        :return: int for hard value, tuple of ints for soft value
        """
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
        """
        A blackjack must be the first two cards, on a non-split hand.
        :return: Boolean indicating whether a hand is blackjack or not
        """
        return len(self) == 2 and self.value() == 21 and not self.is_split

    @property
    def is_splittable(self) -> bool:
        """
        A Hand is splittable if there are only two cards of the same value.
        In some variations, a hand is only splittable if they are of the same rank.
        Ex. a TQ is splittable in our case, but some variations disallow it.
        :return: Boolean indicating whether a hand is splittable or not
        """
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
        """
        A game is finished if the hand is LOST, WON or DRAWn
        :return: Boolean if the status is one of the three above GameState values
        """
        return self.status.value in [GameState.WON, GameState.DRAW, GameState.LOST]

    @property
    def is_idle(self):
        """
        A hand is idle if it is either finished or its last decision is STAND, and hence waiting for the dealer to make their turn.
        The purpose of this method is to help the Game object decide which hand should play next. This hand will be skipped.
        :return: Boolean indicating whether the hand is idle or not
        """
        return self.is_finished or self.status.value == GameState.STAND

    @property
    def profit(self) -> float:
        """
        Returns the profit based on the stake and the state of the hand. The fact that BJ pays 3:2 is hardcoded here.
        :return: Float value of the profit of the hand
        """
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


def format_hand(hand: Hand) -> str | int:
    if hand.is_splittable:
        player_hand = str(hand)

        if player_hand.isnumeric():
            player_hand = int(player_hand)

    elif hand.is_soft_value:
        player_hand = hand.get_string_rep()
    else:
        player_hand = hand.value()

    return player_hand
