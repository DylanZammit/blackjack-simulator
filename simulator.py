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
        assert rank in Card.deck, f'{rank} not found'
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
                raise TypeError(f'Unknown card rank {other}')
        return False

    def __gt__(self, other):
        if isinstance(other, Card):
            return self.value > other.value
        elif isinstance(other, str):
            if other in Card.deck:
                return self.value > Card(other).value
            else:
                raise TypeError(f'Unknown card rank {other}')
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
        if isinstance(card, str):
            card = Card(card)
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
        return max_val

    @property
    def soft_value(self) -> Union[int, tuple[int, int]]:
        if not self.is_soft_value: return self.hard_value
        min_val = self.hard_value + self.num_aces
        max_val = self.hard_value + self.num_aces - 1 + 11
        if 21 in [min_val, max_val]: return 21
        if min_val > 21: return 22
        if max_val > 21: return min_val
        return min_val, max_val

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

    @property
    def profit(self) -> float:
        if self.status == GameState.WON and self.is_blackjack: return 1.5
        if self.status == GameState.WON: return 1
        if self.status == GameState.DRAW: return 0
        if self.status == GameState.LOST: return -1
        return 0

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
                # assert len(hand) == 2, 'All predefined hands must contain exactly 2 cards'
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
            player = self.players[self.player_turn % self.n_players]
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

    def __hit(self, player: Player = None) -> None:
        if player is None: return
        assert player.hand.value != 21, f'Cannot hit on {player.hand}!'

        player.log_decision(GameDecision.HIT)

        card = next(self.draw)
        player.deal_card(card)

        if player.hand.value > 21:
            player.status = GameState.LOST

    def stand(self) -> None:
        return self.__stand(self.next_player())

    def __stand(self, player: Player) -> None:
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

    def play_random(self) -> str:
        player = self.next_player()
        if player is None or player.status == GameState.STAND: return ''

        if player.hand.value <= 11:  # doesn't make sense not to HIT for these hands. Impossible to lose if you hit!
            decision = GameDecision.HIT
        elif player.hand.value == 21:
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

        return decision

    def play_full_random(self) -> None:
        while not self.is_finished:
            self.play_random()

    @property
    def is_finished(self) -> bool:
        return all(player.is_finished for player in self.players)

    @property
    def n_players(self) -> int:
        return len(self.players)

    def get_players_profit(self) -> dict:
        # waaay too complicated!!
        player_names = np.unique([player.name for player in self.players])
        players = {pn: sum(p.profit for p in self.players if p.name == pn) for pn in player_names}
        return players

    def __repr__(self):
        out = ['*' * 30, f'Dealer Hand: {self.dealer.hand.cards} ({self.dealer.hand.value})']

        for player in self.players:
            out.append(f'({player.status}) Player {player.name} Hand: {player.hand.cards} ({player.hand.value})')

        return '\n'.join(out)


def get_best_decision(x: dict):
    hit_exp = x[GameDecision.HIT]['profit'] / x[GameDecision.HIT]['n_occ'] if x[GameDecision.HIT]['n_occ'] > 0 else -10e100
    stand_exp = x[GameDecision.STAND]['profit'] / x[GameDecision.STAND]['n_occ'] if x[GameDecision.STAND]['n_occ'] > 0 else -10e100
    split_exp = x[GameDecision.SPLIT]['profit'] / x[GameDecision.SPLIT]['n_occ'] if x[GameDecision.SPLIT]['n_occ'] > 0 else -10e100
    best_decision = [GameDecision.HIT, GameDecision.STAND, GameDecision.SPLIT][np.argmax([hit_exp, stand_exp, split_exp])]
    expected_profit = max([hit_exp, stand_exp, split_exp])
    return best_decision, expected_profit


def _get_basic_strategy():
    # TODO: run simulations in parallel
    pass


def get_basic_strategy(n_sims: int = 10_000):
    splittable_hands = [f'{c}{c}' for c in 'A23456789T']
    soft_hands = [f'A{c}' for c in 'A23456789'[1:]]
    hard_hands = list(np.arange(5, 22))

    player_starting_vals = hard_hands + splittable_hands + soft_hands
    dealer_starting_vals = list(np.arange(2, 12))
    def_val = {'profit': 0, 'n_occ': 0}
    outcomes = {
        k: {GameDecision.HIT: def_val.copy(), GameDecision.STAND: def_val.copy(), GameDecision.SPLIT: def_val.copy()}
        for k in product(player_starting_vals, dealer_starting_vals)
    }

    for player_val, dealer in product(hard_hands, dealer_starting_vals):
        print(player_val, dealer)
        if isinstance(player_val, str):
            player_hand = ''.join(player_val.split(','))
        else:
            c1 = player_val // 2 + 1
            c2 = player_val // 2 - 1 if player_val % 2 == 0 else player_val // 2
            if c1 == 10: c1 = 'T'
            if c2 == 10: c2 = 'T'
            player_hand = f'{c1}{c2}'

            if player_val == 20: player_hand = '884'
            if player_val == 21: player_hand = '993'

        if dealer == 10:
            dealer_hand = 'T'
        elif dealer == 11:
            dealer_hand = 'A'
        else:
            dealer_hand = str(dealer)

        decisions = []
        if player_val <= 10:  # can you technically infer these using EV?
            decisions = [GameDecision.HIT]
        elif 11 <= player_val < 21:
            decisions = [GameDecision.HIT, GameDecision.STAND]
        elif player_val == 21:
            decisions = [GameDecision.STAND]

        for decision in decisions:
            for _ in range(n_sims):
                game = Blackjack(player_hands=[Hand(player_hand)], dealer_hand=Hand(dealer_hand))
                if decision == GameDecision.HIT: game.hit()
                if decision == GameDecision.STAND: game.stand()

                if not game.is_finished and game.players[0].status != GameState.STAND:
                    game.stand()

                if not game.is_finished:
                    game.hit_dealer()

                players_profit = game.get_players_profit()

                player = game.players[0]

                outcomes[(player_val, dealer)][decision]['profit'] += players_profit[player.name]
                outcomes[(player_val, dealer)][decision]['n_occ'] += 1

    pprint(outcomes)

    best_play = {hands: get_best_decision(outcomes[hands])[0] for hands, _ in outcomes.items()}

    expected_profit = {hands: get_best_decision(outcomes[hands])[1] for hands, _ in outcomes.items()}

    # Could this be neater?
    for soft_hand, dealer in product(soft_hands, dealer_starting_vals):
        hand = Hand(soft_hand)
        hand_val = hand.soft_value
        if isinstance(hand_val, int):
            best_play[(soft_hand, dealer)] = best_play[(hand_val, dealer)]
            expected_profit[(soft_hand, dealer)] = expected_profit[(hand_val, dealer)]
        elif isinstance(hand_val, tuple):
            small_hand, big_hand = hand_val
            big_hand_ev = expected_profit[(big_hand, dealer)]

            if small_hand < 5:
                evs = np.array([expected_profit.get((small_hand + h, dealer), np.nan) for h in range(1, 11)])
                small_hand_ev = np.mean(evs[~np.isnan(evs)])
            else:
                small_hand_ev = expected_profit[(small_hand, dealer)]

            if big_hand_ev > small_hand_ev:
                best_play[(soft_hand, dealer)] = best_play[(big_hand, dealer)]
                expected_profit[(soft_hand, dealer)] = big_hand_ev
            else:
                if small_hand < 5:
                    best_play[(soft_hand, dealer)] = GameDecision.HIT
                    expected_profit[(soft_hand, dealer)] = small_hand_ev
                else:
                    best_play[(soft_hand, dealer)] = best_play[(soft_hand, dealer)]
                    expected_profit[(soft_hand, dealer)] = small_hand_ev

    # could this be neater?
    for splittable_hand, dealer in product(splittable_hands, dealer_starting_vals):
        c = splittable_hand[0]
        if c == 'A': continue
        elif c == '2': continue
        elif c == 'T': card_val = 10
        else: card_val = int(c)

        hand_val = card_val * 2

        no_split_ev = expected_profit[(hand_val, dealer)]
        no_split_bp = best_play[(hand_val, dealer)]

        evs = np.array([expected_profit.get((card_val + h, dealer), np.nan) for h in range(1, 11)])
        split_hand_ev = np.mean(evs[~np.isnan(evs)])

        if split_hand_ev > no_split_ev:
            best_play[(splittable_hand, dealer)] = GameDecision.SPLIT
            expected_profit[(splittable_hand, dealer)] = split_hand_ev
        else:
            best_play[(splittable_hand, dealer)] = no_split_bp
            expected_profit[(splittable_hand, dealer)] = no_split_ev

    df_best_profit = pd.DataFrame(expected_profit.values(), index=list(expected_profit.keys()))
    df_best_profit.index = pd.MultiIndex.from_tuples(df_best_profit.index, names=['player', 'dealer'])
    df_best_profit = df_best_profit.unstack().replace({1: GameDecision.HIT, 0: GameDecision.STAND})

    df_best_play = pd.DataFrame(best_play.values(), index=list(best_play.keys()))
    df_best_play.index = pd.MultiIndex.from_tuples(df_best_play.index, names=['player', 'dealer'])
    df_best_play = df_best_play.unstack().replace({1: GameDecision.HIT, 0: GameDecision.STAND})

    return df_best_play, df_best_profit


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
    # simulate_hand([Hand('23')], Hand('2'))
    df_decision, df_profit = get_basic_strategy(n_sims=1_000)
    pprint(df_decision)
    pprint(df_profit)
    print(df_profit.mean().mean())
