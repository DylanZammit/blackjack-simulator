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
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def card2rank(c):
    if 2 <= c <= 9: return str(c)
    if c == 1 or c == 11: return 'A'
    if c == 10: return 'T'


count_2combinations = {
    i: ['{}{}'.format(card2rank(j), card2rank(i-j))
        for j in range(2, min((i-1)//2+1, 10)) if i-j != j and i-j < 11]
    for i in range(5, 20)
}


@dataclass
class GameDecision:
    SPLIT: str = 'Split'
    STAND: str = 'Stand'
    HIT: str = 'Hit'
    DOUBLE: str = 'Dbl'


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
        if rank == '10':
            rank = 'T'
        elif rank in ['1', '11']:
            rank = 'A'
        else:
            rank = str(rank)

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
    def __init__(self, cards: Union[List[Card], str] = None, is_split: bool = False):

        if cards is None: cards = []

        self.cards = [Card(c) if isinstance(c, str) else c for c in cards]
        self.is_split = is_split

    def __len__(self):
        return len(self.cards)

    def add_card(self, card: Card) -> None:
        if isinstance(card, str):
            card = Card(card)
        self.cards.append(card)

    def __eq__(self, other):
        return self.is_blackjack and other.is_blackjack or \
            not self.is_blackjack and not other.is_blackjack and self.value == other.value

    def __gt__(self, other):
        return self.value > other.value or self.is_blackjack and not other.is_blackjack

    @property
    def value(self) -> int:
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
        return len(self) == 2 and self.value == 21 and not self.is_split

    @property
    def is_splittable(self) -> bool:
        return len(self) == 2 and self.cards[0].value == self.cards[1].value

    @property
    def pretty_val(self):
        return ''.join([c.rank for c in self.cards])

    def __repr__(self):
        return ''.join([c.rank for c in self.cards])


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
        if self.status == GameState.WON and self.is_blackjack: return 1.5 * self.stake
        if self.status == GameState.WON: return self.stake
        if self.status == GameState.DRAW: return 0
        if self.status == GameState.LOST: return -self.stake
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
        self.current_player = self.players[0]

    # TODO: should be a generator... REWORK THIS!!
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

    def double(self) -> None:
        self.double_player(self.next_player())

    def double_player(self, player: Player = None):
        assert player.hand.value != 21, f'Cannot double on {player.hand}!'
        player.stake *= 2
        player.is_doubled = True
        player.log_decision(GameDecision.DOUBLE)
        player.deal_card(next(self.draw))
        player.status = GameState.LOST if player.hand.value > 21 else GameState.STAND

    def hit(self) -> None:
        return self.hit_player(self.next_player())

    def hit_player(self, player: Player = None) -> None:
        if player is None: return
        assert player.hand.value != 21, f'Cannot hit on {player.hand}!'

        player.log_decision(GameDecision.HIT)

        card = next(self.draw)
        player.deal_card(card)

        if player.hand.value > 21:
            player.status = GameState.LOST

    def stand(self) -> None:
        if self.is_dealer_turn: return
        self.stand_player(self.next_player())

    def stand_player(self, player: Player) -> None:
        player.log_decision(GameDecision.STAND)
        player.status = GameState.STAND

    def split(self) -> None:
        return self.split_player(self.next_player())

    def split_player(self, player: Player) -> None:
        if player is None: return
        player.log_decision(GameDecision.SPLIT)

        assert player.hand.is_splittable, f'Cannot Split {player.hand}!'

        player.hand = Hand([player.hand.cards[0]], is_split=True)
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


def simulate_game(
        player_hand: str | int,  # only pass int if you have the value. Should change this
        dealer_hand: int,
        decision: str,
        basic_strategy: dict = None,
        quiet: bool = True,
        n_packs: int = 6,
):

    if isinstance(player_hand, int):
        if player_hand in count_2combinations:
            player_hand = choice(count_2combinations[player_hand])
        elif player_hand == 20: player_hand = '884'
        elif player_hand == 21: player_hand = '993'
        else:
            player_hand = str(player_hand)

    if dealer_hand == 10:
        dh = 'T'
    elif dealer_hand == 11:
        dh = 'A'
    else:
        dh = str(dealer_hand)

    game = Blackjack(
        player_hands=[Hand(player_hand)],
        dealer_hand=Hand(dh),
        n_packs=n_packs,
    )

    if decision == GameDecision.HIT: game.hit()
    if decision == GameDecision.STAND: game.stand()
    if decision == GameDecision.DOUBLE: game.double()
    if decision == GameDecision.SPLIT: game.split()

    # rest can be in a separate method
    while not game.is_dealer_turn:
        player = game.next_player()
        opt_decision = basic_strategy[(player.hand.value, int(dealer_hand))]

        if opt_decision == GameDecision.HIT:
            game.hit_player(player)
        elif opt_decision == GameDecision.STAND:
            game.stand_player(player)
        elif opt_decision == GameDecision.SPLIT:
            game.split_player(player)
        elif opt_decision == GameDecision.DOUBLE:
            game.double_player(player)

    game.hit_dealer()

    players_profit = game.get_players_profit()

    if not quiet:
        print(game)
        print(players_profit)

    return players_profit[game.current_player.name]


def simulate_games_profit(
        n_sims: int,
        player_hand: str | int,
        dealer_hand: int,
        decision: str,
        basic_strategy: dict = None,
        quiet: bool = True,
):
    return sum(simulate_game(player_hand, dealer_hand, decision, basic_strategy, quiet) for _ in range(n_sims))


def get_basic_strategy(n_sims: int = 10_000, n_processes: int = None):
    splittable_hands = [f'{c}{c}' for c in 'A23456789T']
    soft_hands = [f'A{c}' for c in '23456789']
    hard_hands = list([int(x) for x in np.arange(21, 2, -1)])
    blackjack = ['AT']

    player_starting_vals = blackjack + hard_hands + soft_hands + splittable_hands
    dealer_starting_vals = list(np.arange(2, 12))
    outcomes = {k: defaultdict(int) for k in product(player_starting_vals, dealer_starting_vals)}

    best_play, expected_profit = {}, {}

    for player_hand, dealer in product(player_starting_vals, dealer_starting_vals):
        tick = perf_counter()

        player_val = player_hand
        if player_hand == 'T': player_val = 10

        print('Player Hand: {} (Value: {})\tDealer Value: {}....'.format(player_val, player_hand, dealer), end='')

        decisions = []
        if player_hand in splittable_hands:
            decisions = [GameDecision.HIT, GameDecision.STAND, GameDecision.SPLIT, GameDecision.DOUBLE]
        elif player_hand in [21, 'AT']:
            decisions = [GameDecision.STAND]
        elif 'A' in str(player_hand) or 11 <= player_hand < 21:
            decisions = [GameDecision.HIT, GameDecision.STAND, GameDecision.DOUBLE]
        elif player_hand < 11:
            decisions = [GameDecision.HIT, GameDecision.DOUBLE]

        for decision in decisions:

            if n_processes is not None:

                n_processes = mp.cpu_count() if n_processes == -1 else n_processes
                n_batch = n_sims // n_processes + 1

                f = partial(
                    simulate_games_profit,
                    player_hand=player_val,
                    dealer_hand=dealer,
                    decision=decision,
                    basic_strategy=best_play,
                )

                with mp.Pool(n_processes) as pool:
                    res = pool.map(f, [n_batch] * n_processes)
                profit = sum(res)
            else:
                profit = simulate_games_profit(
                    n_sims=n_sims,
                    player_hand=player_val,
                    dealer_hand=dealer,
                    decision=decision,
                    basic_strategy=best_play,
                )

            outcomes[(player_hand, dealer)][decision] = profit
        bp, ev = get_best_decision(outcomes[(player_hand, dealer)], n_sims=n_sims)

        best_play[(player_hand, dealer)] = bp
        expected_profit[(player_hand, dealer)] = ev

        tock = perf_counter()
        print(f'Time taken = {tock - tick:.2f}s')

    pprint(outcomes)

    df_best_profit = pd.DataFrame(expected_profit.values(), index=list(expected_profit.keys()))
    df_best_profit.index = pd.MultiIndex.from_tuples(df_best_profit.index, names=['player', 'dealer'])
    df_best_profit = df_best_profit.unstack()

    df_best_play = pd.DataFrame(best_play.values(), index=list(best_play.keys()))
    df_best_play.index = pd.MultiIndex.from_tuples(df_best_play.index, names=['player', 'dealer'])
    df_best_play = df_best_play.unstack().replace({1: GameDecision.HIT, 0: GameDecision.STAND})

    return df_best_play[0], df_best_profit[0]


def simulate_hand(
        players: str | int,
        dealer: int,
        n_sims=10_000,
        quiet: bool = True,
        basic_strategy: dict = None,
):

    decisions = [GameDecision.STAND, GameDecision.HIT, GameDecision.DOUBLE]
    if isinstance(players, str) and len(players) == 2:
        decisions.append(GameDecision.SPLIT)

    decision_profit = {}
    for decision in decisions:
        print('*'*10 + decision + '*'*10)

        decision_profit[decision] = simulate_games_profit(
            n_sims=n_sims,
            player_hand=players,
            dealer_hand=dealer,
            decision=decision,
            quiet=quiet,
            basic_strategy=basic_strategy,
        )

        print('Expected Profit')
        print(decision, decision_profit[decision] / n_sims)
        print('% {} Wins: {:.2f}%'.format(decision, (1 + decision_profit[decision] / n_sims) / 2 * 100))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Simulate Blackjack games and get optimal basic strategy")

    parser.add_argument("-s", "--samples", type=int, default=None,
                        help="Number of samples per player/dealer hand combination [def: 10_000]")
    parser.add_argument("-p", "--processes", type=int, default=None,
                        help="Number of processes to use (-1 for all). "
                             "NOTE: not beneficial for < 10_000 n_sims [def: no parallel]")
    parser.add_argument("-l", "--load", type=str,
                        help="Path to CSV containing basic strategy to load")
    parser.add_argument("-g", "--generate", action='store_true',
                        help="Generate basic strategy and save")

    args = parser.parse_args()


    if args.load:
        import json
        df = pd.read_csv(args.load, index_col=0)
        json_bs = json.loads(df.to_json())
        bs = {
            (int(player) if player.isnumeric() else player, int(dealer)): decision
            for dealer, player_decision in json_bs.items()
            for player, decision in player_decision.items()
            }
        print(df)
        simulate_hand('44', 6, 1000, quiet=True, basic_strategy=bs)

    if args.samples:

        df_decision, df_profit = get_basic_strategy(
            n_sims=args.samples,
            n_processes=args.processes,
        )

        if args.generate:
            df_decision.to_csv(os.path.join(BASE_PATH, 'basic_strategy.csv'))
            df_profit.to_csv(os.path.join(BASE_PATH, 'basic_strategy_profit.csv'))

        pprint(df_decision)
        pprint(df_profit)
        print(df_profit.mean().mean())
