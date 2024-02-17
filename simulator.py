from copy import deepcopy
from typing import Union, List
from random import choice, shuffle
from pprint import pprint
import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
from dataclasses import dataclass
import multiprocessing


@dataclass
class GameDecision:
    SPLIT: str = 'SP'
    STAND: str = 'S'
    HIT: str = 'H'


@dataclass
class GameState:
    WON: str = 'WON'
    LOST: str = 'LOST'
    DRAW: str = 'DRAW'
    LIVE: str = 'LIVE'
    STAND: str = 'STAND'


class Card:

    deck = 'A23456789TJQK'

    def __init__(self, rank: str):
        assert rank in Card.deck
        self.rank = rank

    def __repr__(self):
        return self.rank

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.value == other.value
        elif isinstance(other, str):
            if other in Card.deck:
                return self.value == Card(other).value
            else:
                raise TypeError(f'Unkonw card rank {other}')
        return False

    def __gt__(self, other):
        if isinstance(other, Card):
            return self.value > other.value
        elif isinstance(other, str):
            if other in Card.deck:
                return self.value > Card(other).value
            else:
                raise TypeError(f'Unkonw card rank {other}')
        else:
            raise TypeError(f'Cannot add {type(other)}')

    @property
    def value(self) -> Union[int, tuple[int, int]]:
        if self.rank == 'A': return 1, 11
        if self.rank in 'TJQK': return 10
        return int(self.rank)


class Hand:
    def __init__(self, cards: Union[List[Card], str] = None):

        if cards is None: cards = []

        self.cards = [Card(c) if isinstance(c, str) else c for c in cards]

    def __len__(self):
        return len(self.cards)

    def add_card(self, card: Card) -> None:
        self.cards.append(card)

    def __eq__(self, other):
        if self.is_blackjack and other.is_blackjack:
            return True

        if not self.is_blackjack and not other.is_blackjack and self.value == other.value:
            return True

        return False

    def __gt__(self, other):
        return self.value > other.value or self.is_blackjack and not other.is_blackjack

    @property
    def value(self) -> Union[int, tuple[int, int]]:
        if not self.is_soft_value: return self.hard_value
        min_val = self.hard_value + self.num_aces
        max_val = self.hard_value + self.num_aces - 1 + 11
        if 21 in [min_val, max_val]: return 21
        if min_val > 21: return 22
        if max_val > 21: return min_val
        return min_val#, max_val

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
        return Card('A') in self.cards

    @property
    def is_blackjack(self) -> bool:
        return len(self) == 2 and self.value == 21

    @property
    def is_splittable(self) -> bool:
        return len(self) == 2 and self.cards[0].value == self.cards[1].value


class Player:
    def __init__(
            self,
            name: Union[str, int],
            hand: Hand = None,
    ):
        self.name = name
        self.hand = hand if hand is not None else Hand()
        self.status = GameState.LIVE
        self.decision_hist = []

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

    def __repr__(self):
        return f'Player {self.name}: hand={self.hand} ({self.hand.value}) [{self.status}]'


class Blackjack:

    def __init__(
            self,
            n_packs: int = 4,
            n_players: int = 1,
            player_hands: List[Hand] = None,
            dealer_hand: Hand = None,
            blackjack_payout: float = 1.5,
    ):
        cards = 'A23456789TJQK'
        self.blackjack_payout = blackjack_payout
        self.n_packs = n_packs
        self.shoe = list(cards) * n_packs * 4
        shuffle(self.shoe)
        self.dealer = Player('dealer')
        self.players = [Player(i) for i in range(n_players)]

        self.draw = self.draw_card()

        if player_hands is not None:
            assert len(player_hands) == n_players, 'There must be predefined hands as there are players'
            assert dealer_hand is not None, 'Must also specify the dealer\'s hand if the players\' hand is specified'

            for player, hand in zip(self.players, player_hands):
                assert len(hand) == 2, 'All predefined hands must contain exactly 2 cards'
                player.hand = hand

            self.dealer.hand = dealer_hand

            if len(dealer_hand) == 1:
                self.dealer.deal_card(next(self.draw))

        else:
            self.__setup()

        self.player_turn = 0

    def next_player(self) -> Union[Player, None]:

        if all([player.is_finished or player.status == GameState.STAND for player in self.players]):
            self.hit_dealer()
            return

        is_finished = True
        player = None
        while is_finished:
            if self.player_turn == self.n_players:
                self.player_turn = 0
            player = self.players[self.player_turn]
            is_finished = player.is_finished
            self.player_turn += 1

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

        while self.dealer.hand.value <= 16 or self.dealer.hand.value == 17 and self.dealer.hand.is_soft_value:
            self.dealer.deal_card(next(self.draw))

        for player in self.players:
            if player.is_finished: continue
            if self.dealer.hand.value > 21:
                player.status = GameState.WON
            elif self.dealer.hand < player.hand:
                player.status = GameState.WON
            elif self.dealer.hand > player.hand:
                player.status = GameState.LOST
            elif self.dealer.hand == player.hand:
                player.status = GameState.DRAW
        # self.dealer.status = GameState.GAME_OVER

    def hit(self) -> None:
        return self.__hit(self.next_player())

    def __hit(self, player=None) -> None:
        if player is None: return
        assert player.hand.value != 21, f'Cannot hit on {player.hand}!'

        player.log_decision(GameDecision.HIT)

        card = next(self.draw)
        player.deal_card(card)

        if player.hand.value > 21:
            player.status = GameState.LOST

    def stand(self) -> None:
        return self.__stand(self.next_player())

    def __stand(self, player) -> None:
        player.log_decision(GameDecision.STAND)
        player.status = GameState.STAND

    def split(self) -> None:
        return self.__split(self.next_player())

    def __split(self, player: Player) -> None:
        if player is None: return
        player.log_decision(GameDecision.SPLIT)

        assert player.hand.is_splittable, f'Cannot Split {player.hand}!'

        player.hand = Hand([player.hand.cards[0]])
        new_player = Player(name=player.name, hand=deepcopy(player.hand))

        self.players.insert(self.player_turn+1, new_player)
        player.deal_card(next(self.draw))
        new_player.deal_card(next(self.draw))

        self.player_turn += 1  # account for new hand

    def play_random(self) -> None:
        player = self.next_player()
        if player is None or player.status == GameState.STAND: return

        if player.hand.value == 21:
            decision = GameDecision.STAND
        elif player.hand.is_splittable:
            decision = choice([GameDecision.STAND, GameDecision.HIT, GameDecision.SPLIT])
        else:
            decision = choice([GameDecision.STAND, GameDecision.HIT])

        if decision == GameDecision.HIT:
            self.__hit(player)
        elif decision == GameDecision.STAND:
            self.__stand(player)
        elif decision == GameDecision.SPLIT:
            self.__split(player)

    def play_full_random(self) -> None:
        while not self.is_finished:
            self.play_random()

    @property
    def is_finished(self) -> bool:
        return all(player.is_finished for player in self.players)

    @property
    def n_players(self) -> int:
        return len(self.players)

    def get_players_profit(self) -> defaultdict:
        players = defaultdict(int)

        for player in self.players:
            if player.status == GameState.WON and player.is_blackjack:
                player.payout = self.blackjack_payout
            elif player.status == GameState.WON:
                player.payout = 1
            elif player.status == GameState.DRAW:
                player.payout = 0
            elif player.status == GameState.LOST:
                player.payout = -1

            players[player.name] += player.payout

        return players

    def __repr__(self):
        out = ['*' * 30, f'Dealer Hand: {self.dealer.hand.cards} ({self.dealer.hand.value})']

        for player in self.players:
            out.append(f'({player.status}) Player {player.name} Hand: {player.hand.cards} ({player.hand.value})')

        return '\n'.join(out)


def get_best_decision(x: dict):
    hit_exp = x[GameDecision.HIT]['profit'] / x[GameDecision.HIT]['n_occ']# if x[GameDecision.HIT]['n_occ'] > 0 else -10e100
    stand_exp = x[GameDecision.STAND]['profit'] / x[GameDecision.STAND]['n_occ']# if x[GameDecision.STAND]['n_occ'] > 0 else -10e100
    split_exp = x[GameDecision.SPLIT]['profit'] / x[GameDecision.SPLIT]['n_occ'] if x[GameDecision.SPLIT]['n_occ'] > 0 else -10e100
    return [GameDecision.HIT, GameDecision.STAND, GameDecision.SPLIT][np.argmax([hit_exp, stand_exp, split_exp])]


def _get_basic_strategy():
    pass


def get_basic_strategy(n_sims: int = 100_000):

    player_starting_vals = list(np.arange(5, 22)) + [f'{c},{c}' for c in 'A23456789T'] + [f'A,{c}' for c in 'A23456789T'[1:]]
    dealer_starting_vals = range(2, 12)
    def_val = {'profit': 0, 'n_occ': 0}
    outcomes = {
        k: {GameDecision.HIT: def_val.copy(), GameDecision.STAND: def_val.copy(), GameDecision.SPLIT: def_val.copy()}
        for k in product(player_starting_vals, dealer_starting_vals)
    }

    for i in range(1, n_sims + 1):
        if i % 5_000 == 0: print(f'{int(i/n_sims*100)}%')
        game = Blackjack(n_players=10)

        game.play_full_random()

        dealer_first_card = 11 if game.dealer.hand.cards[0] == 'A' else game.dealer.hand.cards[0].value

        players_profit = game.get_players_profit()
        for player in game.players:
            hand, decision = player.decision_hist[-1]
            split_hand = next((hand for hand, decision in player.decision_hist[::-1] if decision == GameDecision.SPLIT), None)

            if split_hand is not None:
                c = 'T' if split_hand.cards[0].rank in 'JQK' else split_hand.cards[0]
                key_split = '{card},{card}'.format(card=c)
                decision_split = GameDecision.SPLIT

                outcomes[(key_split, dealer_first_card)][decision_split]['profit'] += players_profit[player.name]
                outcomes[(key_split, dealer_first_card)][decision_split]['n_occ'] += 1

            if hand.is_splittable:
                c1 = 'T' if hand.cards[0].rank in 'JQK' else hand.cards[0]
                c2 = 'T' if hand.cards[1].rank in 'JQK' else hand.cards[1]
                key = f'{c1},{c2}'
            elif hand.is_soft_value:
                c = hand.cards[0] if hand.cards[1].rank == 'A' else hand.cards[1]
                c = 'T' if c.rank in 'JQK' else c
                key = f'A,{c}'
            else:
                key = hand.value

            outcomes[(key, dealer_first_card)][decision]['profit'] += players_profit[player.name]
            outcomes[(key, dealer_first_card)][decision]['n_occ'] += 1

    pprint(outcomes)

    best_play = {hands: get_best_decision(outcomes[hands]) for hands, _ in outcomes.items()}
    df_best_play = pd.DataFrame(best_play.values(), index=list(best_play.keys()))
    df_best_play.index = pd.MultiIndex.from_tuples(df_best_play.index, names=['player', 'dealer'])
    df_best_play = df_best_play.unstack().replace({1: GameDecision.HIT, 0: GameDecision.STAND})

    return df_best_play


def simulate_hand(players: List[Hand], dealer: Hand, n_sims=10_000):
    res = {GameDecision.HIT: 0, GameDecision.STAND: 0}

    for i in range(n_sims):
        game = Blackjack(
            n_packs=4,
            n_players=len(players),
            player_hands=deepcopy(players),
            dealer_hand=deepcopy(dealer),
        )
        game.play_full_random()
        decision = game.players[0].decision_hist[-1][1]
        status = game.players[0].status
        print(game)
        print(decision, status)

        res[decision] += game.get_players_profit()[0]

    print('Expected Profit')
    print(GameDecision.HIT, res[GameDecision.HIT] / n_sims)
    print(GameDecision.STAND, res[GameDecision.STAND] / n_sims)


if __name__ == '__main__':
    simulate_hand([Hand('23')], Hand('2'))
    df = get_basic_strategy(n_sims=10_000)
    pprint(df)
