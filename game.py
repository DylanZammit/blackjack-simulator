from utils import GameDecision, GameState
from hand import Hand
from card import Card
from player import Player
from typing import Union, List
from copy import deepcopy
from random import shuffle, choice
import numpy as np


class Blackjack:

    def __init__(
            self,
            n_packs: int = 4,
            n_players: int = 1,
            player_hands: List[Hand] = None,
            dealer_hand: Hand = None,
            blackjack_payout: float = 1.5,
            hit_on_soft_17: bool = True,
            double_after_split: bool = False,
            quiet: bool = True,
    ):
        cards = 'A23456789TJQK'
        self.hit_on_17 = hit_on_soft_17
        self.blackjack_payout = blackjack_payout
        self.n_packs = n_packs
        self.shoe = list(cards) * n_packs * 4
        self.double_after_split = double_after_split
        shuffle(self.shoe)
        self.dealer = Player('dealer')
        self.players = [Player(i) for i in range(n_players)]

        self.draw = self.draw_card()
        self.quiet = quiet

        if player_hands is not None:
            assert len(player_hands) == n_players, 'There must be predefined hands as there are players'
            assert dealer_hand is not None, 'Must also specify the dealer\'s hand if the players\' hand is specified'

            for player, hand in zip(self.players, player_hands):
                player.hand = hand

                # remove cards from shoe
                for card in hand.cards:
                    self.shoe.remove(str(card))

            self.dealer.hand = dealer_hand

            for card in dealer_hand.cards:
                self.shoe.remove(str(card))

            if len(dealer_hand) == 1:
                self.dealer.deal_card(next(self.draw))

        else:
            self.__setup()

        if self.dealer.is_blackjack:
            for player in self.players:
                self.stand_player(player)

            self.hit_dealer()

        self.player_turn = 0
        self.current_player = self.players[0]

    # TODO: should be a generator... REWORK THIS!!
    # TODO: should give same player if no action has been done
    def next_player(self) -> Union[Player, None]:

        if all([player.is_finished or player.status == GameState.STAND for player in self.players]):
            self.hit_dealer()
            return

        is_finished = True
        player = None
        while is_finished:
            player = self.players[self.player_turn % self.n_players]
            is_finished = player.is_finished or player.status == GameState.STAND
            self.player_turn += 1
        self.current_player = player
        return player

    def draw_card(self):
        while not self.is_finished:
            yield Card(self.shoe.pop())
        yield -1

    def __setup(self) -> None:

        for player in self.players:
            player.deal_card(next(self.draw))
            player.deal_card(next(self.draw))

        self.dealer.deal_card(next(self.draw))
        self.dealer.deal_card(next(self.draw))

    def hit_dealer(self) -> None:
        if not self.is_dealer_turn:
            if not self.quiet:
                print('It is not the dealer\'s turn!')
            return

        if self.is_finished:
            if not self.quiet:
                print('Game is already finished!')
            return

        while self.dealer.hand.value <= 16 or \
                self.dealer.hand.value == 17 and self.dealer.hand.is_soft_value and self.hit_on_17:
            self.dealer.deal_card(next(self.draw))

        for player in self.players:
            if player.is_finished:
                continue
            if self.dealer.hand.value > 21:
                player.status = GameState.WON
            elif self.dealer.hand < player.hand:
                player.status = GameState.WON
            elif self.dealer.hand > player.hand:
                player.status = GameState.LOST
            elif self.dealer.hand == player.hand:
                player.status = GameState.DRAW

    def double(self) -> None:
        self.double_player(self.next_player())

    def double_player(self, player: Player = None):
        assert player.hand.value != 21, f'Cannot double on {player.hand}!'
        assert len(player.hand) == 2, f'Can only double on starting hands, current hand: {player.hand}'
        assert self.double_after_split or not player.hand.is_split, 'Cannot double after split!'

        player.stake *= 2
        player.is_doubled = True
        player.log_decision(GameDecision.DOUBLE)
        player.deal_card(next(self.draw))
        player.status = GameState.LOST if player.hand.value > 21 else GameState.STAND

    def hit(self) -> None:
        return self.hit_player(self.next_player())

    def hit_player(self, player: Player = None) -> None:
        if player is None:
            return

        assert player.hand.value != 21, f'Cannot hit on {player.hand}!'

        player.log_decision(GameDecision.HIT)

        card = next(self.draw)
        player.deal_card(card)

        if player.hand.value > 21:
            player.status = GameState.LOST

    def stand(self) -> None:
        if self.is_dealer_turn:
            return
        self.stand_player(self.next_player())

    @staticmethod
    def stand_player(player: Player) -> None:
        if player.is_finished:
            return
        player.log_decision(GameDecision.STAND)
        player.status = GameState.STAND

    def split(self) -> None:
        return self.split_player(self.next_player())

    def split_player(self, player: Player) -> None:
        if player is None:
            return
        player.log_decision(GameDecision.SPLIT)

        assert player.hand.is_splittable, f'Cannot Split {player.hand}!'

        player.hand = Hand([player.hand.cards[0]], is_split=True)
        new_player = Player(name=player.name, hand=deepcopy(player.hand))

        self.players.insert(self.player_turn + 1, new_player)
        player.deal_card(next(self.draw))
        new_player.deal_card(next(self.draw))

        self.player_turn += 1  # account for new hand

    def play_random(self) -> str:
        player = self.next_player()
        if player is None or player.status == GameState.STAND:
            return ''

        if player.hand.value <= 11:  # doesn't make sense not to HIT for these hands. Impossible to lose if you hit!
            decision = GameDecision.HIT
        elif player.hand.value == 21:
            decision = GameDecision.STAND
        elif player.hand.is_splittable:
            decision = choice([GameDecision.STAND, GameDecision.HIT, GameDecision.SPLIT, GameDecision.DOUBLE])
        else:
            decision = choice([GameDecision.STAND, GameDecision.HIT, GameDecision.DOUBLE])

        if decision == GameDecision.HIT:
            self.hit_player(player)
        elif decision == GameDecision.STAND:
            self.stand_player(player)
        elif decision == GameDecision.SPLIT:
            self.split_player(player)
        elif decision == GameDecision.DOUBLE:
            self.double_player(player)

        return decision

    def play_full_random(self) -> None:
        while not self.is_finished:
            self.play_random()

    @property
    def is_finished(self) -> bool:
        return all(player.is_finished for player in self.players)

    @property
    def is_dealer_turn(self) -> bool:
        all_players_standing = all(
            player.status == GameState.STAND for player in self.players if not player.is_finished)
        return self.is_finished or all_players_standing

    @property
    def n_players(self) -> int:
        return len(self.players)

    # TODO: find a neater way to do this
    def get_players_profit(self) -> dict:
        player_names = np.unique([player.name for player in self.players])
        players = {pn: sum(p.profit for p in self.players if p.name == pn) for pn in player_names}
        return players

    def __repr__(self):
        out = ['*' * 30, f'Dealer Hand: {self.dealer.hand.cards} ({self.dealer.hand.value})']

        for player in self.players:
            out.append(f'({player.status}) Player {player.name} Hand: {player.hand.cards} ({player.hand.value})')

        return '\n'.join(out)
