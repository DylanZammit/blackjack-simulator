from collections import defaultdict
from random import choice, shuffle
from dataclasses import dataclass
from typing import Union, List
from itertools import product
from functools import partial
from time import perf_counter
import multiprocessing as mp
from copy import deepcopy
from pprint import pprint
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


@dataclass
class GameDecision:
    SPLIT: str = 'Split'
    STAND: str = 'Stand'
    HIT: str = 'Hit'


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
        if self.num_aces == 0: return self.hard_value
        min_val = self.hard_value + self.num_aces
        max_val = self.hard_value + self.num_aces - 1 + 11
        if 21 in [min_val, max_val]: return 21
        if min_val > 21: return min_val
        if max_val > 21: return min_val
        return max_val

    @property
    def soft_value(self) -> Union[int, tuple[int, int]]:
        if self.num_aces == 0: return self.hard_value
        min_val = self.hard_value + self.num_aces
        max_val = self.hard_value + self.num_aces - 1 + 11
        if 21 in [min_val, max_val]: return 21
        if min_val > 21: return min_val
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
        return isinstance(self.soft_value, tuple)

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
            quiet: bool = True,
    ):
        cards = 'A23456789TJQK'
        self.blackjack_payout = blackjack_payout
        self.n_packs = n_packs
        self.shoe = list(cards) * n_packs * 4
        shuffle(self.shoe)
        self.dealer = Player('dealer')
        self.players = [Player(i) for i in range(n_players)]

        self.draw = self.draw_card()
        self.quiet = quiet

        if player_hands is not None:
            assert len(player_hands) == n_players, 'There must be predefined hands as there are players'
            assert dealer_hand is not None, 'Must also specify the dealer\'s hand if the players\' hand is specified'

            for player, hand in zip(self.players, player_hands):
                # assert len(hand) == 2, 'All predefined hands must contain exactly 2 cards'
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
        if not self.is_dealer_turn:
            if not self.quiet: print('It is not the dealer\'s turn!')
            return

        if self.is_finished:
            if not self.quiet: print('Game is already finished!')
            return

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
        if self.is_dealer_turn: return
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
    def is_dealer_turn(self) -> bool:
        all_players_standing = all(player.status == GameState.STAND for player in self.players if not player.is_finished)
        return self.is_finished or all_players_standing

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


def get_best_decision(x: dict, n_sims: int):
    if not len(x): return GameDecision.HIT, 0

    return max(x, key=x.get), max(x.values()) / n_sims


def simulate_game(player_hand: str, dealer_hand: str, decision: str, quiet: bool = True):
    game = Blackjack(player_hands=[Hand(player_hand)], dealer_hand=Hand(dealer_hand))
    if decision == GameDecision.HIT: game.hit()
    if decision == GameDecision.STAND: game.stand()
    game.stand()
    game.hit_dealer()

    if not quiet: print(game)

    players_profit = game.get_players_profit()

    player = game.players[0]
    return players_profit[player.name]


def simulate_games_profit(
        n_sims: int,
        player_hand: str,
        dealer_hand: str,
        decision: str,
        quiet: bool = True,
):
    return sum(simulate_game(player_hand, dealer_hand, decision, quiet) for _ in range(n_sims))


def get_basic_strategy(n_sims: int = 10_000, n_processes: int = None):
    splittable_hands = [f'{c}{c}' for c in 'A23456789T']
    soft_hands = [f'A{c}' for c in '23456789']
    small_hands = list(np.arange(3, 11))
    hard_hands = list(np.arange(11, 22))
    blackjack = ['AT']

    player_starting_vals = small_hands + hard_hands + splittable_hands + soft_hands + blackjack
    dealer_starting_vals = list(np.arange(2, 12))
    outcomes = {k: defaultdict(int) for k in product(player_starting_vals, dealer_starting_vals)}

    for player_val, dealer in product(blackjack + hard_hands + soft_hands, dealer_starting_vals):
        tick = perf_counter()

        player_hand = player_val
        if not isinstance(player_val, str):
            c1 = player_val // 2 + 1
            c2 = player_val // 2 - 1 if player_val % 2 == 0 else player_val // 2
            if c1 == 10: c1 = 'T'
            if c2 == 10: c2 = 'T'
            player_hand = f'{c1}{c2}'

            if player_val == 20: player_hand = '884'
            if player_val == 21: player_hand = '993'

        print('Player Hand: {} (Value: {})\tDealer Value: {}....'.format(player_hand, player_val, dealer), end='')
        if dealer == 10:
            dealer_hand = 'T'
        elif dealer == 11:
            dealer_hand = 'A'
        else:
            dealer_hand = str(dealer)

        decisions = []
        if player_val in [21, 'AT']:
            decisions = [GameDecision.STAND]
        elif 'A' in str(player_val) or 11 <= player_val < 21:
            decisions = [GameDecision.HIT, GameDecision.STAND]

        for decision in decisions:

            if n_processes is not None:

                n_processes = mp.cpu_count() if n_processes == -1 else n_processes
                n_batch = n_sims // n_processes + 1

                f = partial(simulate_games_profit, player_hand=player_hand, dealer_hand=dealer_hand, decision=decision)
                with mp.Pool(n_processes) as pool:
                    res = pool.map(f, [n_batch] * n_processes)
                profit = sum(res)
            else:
                profit = simulate_games_profit(
                    n_sims=n_sims,
                    player_hand=player_hand,
                    dealer_hand=dealer_hand,
                    decision=decision,
                )

            outcomes[(player_val, dealer)][decision] = profit

        tock = perf_counter()
        print(f'Time taken = {tock - tick:.2f}s')

    pprint(outcomes)

    best_play = {}
    expected_profit = {}
    for hands in outcomes:
        bp, ev = get_best_decision(outcomes[hands], n_sims=n_sims)
        best_play[hands] = bp
        expected_profit[hands] = ev

    for small_hand, dealer in product(small_hands[::-1], dealer_starting_vals):
        vals = list(range(2, 10)) + [10] * 4
        ev_1 = expected_profit[(small_hand + 1, dealer)]
        ev_11 = expected_profit[(small_hand + 11, dealer)]
        evs = np.array([expected_profit[(small_hand + h, dealer)] for h in vals] + [max([ev_1, ev_11])])
        small_hand_ev = np.mean(evs)

        best_play[(small_hand, dealer)] = GameDecision.HIT  # best play is obviously HIT
        expected_profit[(small_hand, dealer)] = small_hand_ev

    # Could this be neater?
    # Calculating EV of soft hands
    # E[AX] = max(E[11 + X], E[1 + X])
    if False:
        for soft_hand, dealer in product(soft_hands, dealer_starting_vals):
            hand = Hand(soft_hand)
            hand_val = hand.soft_value
            # if isinstance(hand_val, int):
            if not hand.is_soft_value:
                best_play[(soft_hand, dealer)] = best_play[(hand_val, dealer)]
                expected_profit[(soft_hand, dealer)] = expected_profit[(hand_val, dealer)]
            else:
            # elif isinstance(hand_val, tuple):
                small_hand, big_hand = hand_val
                big_hand_ev = expected_profit[(big_hand, dealer)]
                small_hand_ev = expected_profit[(small_hand, dealer)]

                if big_hand_ev > small_hand_ev:
                    best_play[(soft_hand, dealer)] = best_play[(big_hand, dealer)]
                    expected_profit[(soft_hand, dealer)] = big_hand_ev
                else:
                    best_play[(soft_hand, dealer)] = best_play[(soft_hand, dealer)]
                    expected_profit[(soft_hand, dealer)] = small_hand_ev

    # could this be neater?
    # Calculating EV of splittable hands
    # E[XX] = max(E[X + X], 2 * E[X]) = max(E[X + X], 2 * Sum_[i in A2..JQK](E[X + i])
    for splittable_hand, dealer in product(splittable_hands, dealer_starting_vals):
        c = splittable_hand[0]
        if c == 'A': card_val = 6
        elif c == 'T': card_val = 10
        else: card_val = int(c)

        hand_val = card_val * 2

        no_split_ev = expected_profit[(hand_val, dealer)]
        no_split_bp = best_play[(hand_val, dealer)]

        if c == 'A':
            vals = list(range(2, 10))  # + [10] * 4
            evs = np.array([expected_profit[('A' + str(h), dealer)] for h in vals])
            for _ in range(4):
                evs = np.insert(evs, 0, expected_profit[(21, dealer)])  # split AT NOT BJ
            evs = np.insert(evs, 0, expected_profit[(12, dealer)])  # handle AA separately
        else:
            vals = list(range(2, 10)) + [10] * 4
            evs = np.array([expected_profit[(card_val + h, dealer)] for h in vals])

            # handle AT separately
            if card_val == 10:
                evs = np.insert(evs, 0, expected_profit[(21, dealer)])  # split AT NOT BJ
            else:
                evs = np.insert(evs, 0, expected_profit[(f'A{card_val}', dealer)])

        split_hand_ev = 2 * np.mean(evs[~np.isnan(evs)])

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


def simulate_hand(players: str, dealer: str, n_sims=10_000, quiet: bool = True):

    print('*'*10 + GameDecision.HIT + '*'*10)

    hit_profit = simulate_games_profit(
        n_sims=n_sims,
        player_hand=players,
        dealer_hand=dealer,
        decision=GameDecision.HIT,
        quiet=quiet,
    )

    print('*'*10 + GameDecision.STAND + '*'*10)

    stand_profit = simulate_games_profit(
        n_sims=n_sims,
        player_hand=players,
        dealer_hand=dealer,
        decision=GameDecision.STAND,
        quiet=quiet,
    )

    print('Expected Profit')
    print(GameDecision.HIT, hit_profit / n_sims)
    print(GameDecision.STAND, stand_profit / n_sims)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Simulate Blackjack games and get optimal basic strategy")

    parser.add_argument("-s", "--samples", type=int, default=1_000,
                        help="Number of samples per player/dealer hand combination [def: 10_000]")
    parser.add_argument("-p", "--processes", type=int, default=None,
                        help="Number of processes to use (-1 for all) [def: no parallel]")

    args = parser.parse_args()

    # simulate_hand('A7', 'A', 10000, quiet=False)
    # simulate_hand('78', 'A', 10000, quiet=False)

    if True:
        # run parallel not beneficial for < 10_000 n_sims
        df_decision, df_profit = get_basic_strategy(n_sims=args.samples, n_processes=args.processes)
        pprint(df_decision)
        pprint(df_profit)
        print(df_profit.mean().mean())
