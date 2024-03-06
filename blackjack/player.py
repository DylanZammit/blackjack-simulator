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
        """
        A Player can have many Hands each having many Cards
        :param name: Pretty name of the player, can be either string or int
        :param hand: The initial Hand object the player has.
        :param stake: The initial stake the player has for their initial hand
        """
        self.name = name
        hand = hand if hand is not None else Hand(stake=stake, player_name=name)
        self.hands = [hand]
        self.decision_hist = []
        self.original_stake = stake

    def reset(self):
        self.hands = [Hand(stake=self.original_stake, player_name=self.name)]

    @property
    def hand(self) -> Hand:
        """
        :return: The first hand the player was dealt. Particularly useful for the dealer, since they can only have one hand at most
        """
        return self.hands[0]

    @property
    def status(self) -> GameState:
        """
        :return: A GameState object describing whether the player is still LIVE or finished the game in a WIN/DRAW/LOST.
        """
        if any(not hand.is_finished for hand in self.hands) or not len(self.hands):
            return GameState.LIVE

        if self.profit == 0:
            return GameState.DRAW
        elif self.profit > 0:
            return GameState.WON
        elif self.profit < 0:
            return GameState.LOST

    @property
    def profit(self) -> float:
        """
        :return: The total (sum) of profits of all hands of the player. Negative value for loss
        """
        return sum(hand.profit for hand in self.hands)

    @property
    def stake(self) -> int:
        """
        :return: The total (sum) of stake of all hands of the player.
        """
        return sum(hand.stake for hand in self.hands)

    @property
    def is_finished(self) -> bool:
        """
        :return: Boolean indicating whether all player hands are finished (WIN/DRAW/LOST)
        """
        return all(hand.is_finished for hand in self.hands)

    def __repr__(self):
        return f'Player {self.name}: hands={self.hands} [{self.status}]'
