from blackjack.utils import GameDecision, count_2combinations, DATA_PATH
from collections import defaultdict
from itertools import product
from functools import partial
from time import perf_counter
import multiprocessing as mp
from pprint import pprint
from random import choice

from blackjack.game import Blackjack
from blackjack.hand import Hand

import pandas as pd
import numpy as np
import json
import os


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def get_best_decision(x: dict, n_sims: int):
    if not len(x):
        return GameDecision.HIT, 0

    return max(x, key=x.get), max(x.values()) / n_sims


def simulate_game(
        player_hand: str | int,  # only pass int if you have the value. Should change this
        dealer_hand: int,
        decision: str,
        basic_strategy: dict = None,
        expected_profit: dict = None,
        quiet: bool = True,
        n_packs: int = 6,
        double_after_split: bool = True,
        hit_on_soft_17: bool = True,
):

    if isinstance(player_hand, int):
        if player_hand in count_2combinations:
            player_hand = choice(count_2combinations[player_hand])
        elif player_hand == 20:
            player_hand = '884'
        elif player_hand == 21:
            player_hand = '993'
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
        double_after_split=double_after_split,
        hit_on_soft_17=hit_on_soft_17,
    )

    if not game.is_finished:

        if decision == GameDecision.HIT:
            game.hit()
        if decision == GameDecision.STAND:
            game.stand()
        if decision == GameDecision.DOUBLE:
            game.double()
        if decision == GameDecision.SPLIT:
            game.split()

        # rest can be in a separate method
        while not game.is_dealer_turn:
            player = game.next_player()

            if isinstance(player.hand.soft_value, tuple):
                # Check whether to consider low or high soft value decision based on EV
                min_val, max_val = player.hand.soft_value

                ev_min = expected_profit[(min_val, dealer_hand)]
                ev_max = expected_profit[(max_val, dealer_hand)]

                val = min_val if ev_min > ev_max else max_val
                opt_decision = basic_strategy[(val, dealer_hand)]
            elif decision == GameDecision.SPLIT and player.hand.get_string_rep() == player_hand:
                # Always resplit if you think it is optimal
                opt_decision = GameDecision.SPLIT
            else:
                opt_decision = basic_strategy[(player.hand.value, dealer_hand)]

            # Can only double on first two cards. Hit otherwise
            if opt_decision == GameDecision.DOUBLE and len(player.hand) != 2:
                opt_decision = GameDecision.HIT

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
        for player in game.players:
            print(f'Player {player.name}:', player.decision_hist)

    return players_profit[game.current_player.name]


def simulate_games_profit(
        n_sims: int,
        player_hand: str | int,
        dealer_hand: int,
        decision: str,
        basic_strategy: dict = None,
        expected_profit: dict = None,
        double_after_split: bool = True,
        hit_on_soft_17: bool = True,
        num_decks: bool = 6,
        quiet: bool = True,
):
    return sum(simulate_game(
        player_hand=player_hand,
        dealer_hand=dealer_hand,
        decision=decision,
        basic_strategy=basic_strategy,
        expected_profit=expected_profit,
        double_after_split=double_after_split,
        hit_on_soft_17=hit_on_soft_17,
        n_packs=num_decks,
        quiet=quiet,
    ) for _ in range(n_sims))


def get_basic_strategy(
        n_sims: int = 10_000, 
        n_processes: int = None,
        double_after_split: bool = True,
        hit_on_soft_17: bool = True,
        num_decks: bool = 6,
):
    splittable_hands = [f'{c}{c}' for c in 'A23456789T']
    soft_hands = [f'A{c}' for c in '23456789']
    hard_hands = list([int(x) for x in np.arange(21, 1, -1)])
    blackjack = ['AT']

    # noinspection PyTypeChecker
    player_starting_vals = blackjack + hard_hands + soft_hands + splittable_hands
    dealer_starting_vals = list(np.arange(2, 12))
    outcomes = {k: defaultdict(int) for k in product(player_starting_vals, dealer_starting_vals)}

    best_play, expected_profit = {}, {}

    for player_hand, dealer in product(player_starting_vals, dealer_starting_vals):
        tick = perf_counter()

        player_val = player_hand
        if player_hand == 'T':
            player_val = 10

        print('Player Hand: {} (Value: {})\tDealer Value: {}....'.format(player_val, player_hand, dealer), end='')

        decisions = []
        if player_hand in splittable_hands:
            decisions = [GameDecision.HIT, GameDecision.STAND, GameDecision.SPLIT, GameDecision.DOUBLE]
        elif player_hand in [21, 'AT']:
            decisions = [GameDecision.STAND]
        elif 'A' in str(player_hand) or 11 <= player_hand < 20:
            decisions = [GameDecision.HIT, GameDecision.STAND, GameDecision.DOUBLE]
        elif 5 <= player_hand < 11:
            decisions = [GameDecision.HIT, GameDecision.DOUBLE]
        elif player_hand == 20 or player_hand < 5:
            decisions = [GameDecision.HIT, GameDecision.STAND]

        for decision in decisions:

            simulate_games_profit_partial = partial(
                simulate_games_profit,
                player_hand=player_val,
                dealer_hand=dealer,
                decision=decision,
                basic_strategy=best_play,
                expected_profit=expected_profit,
                double_after_split=double_after_split,
                hit_on_soft_17=hit_on_soft_17,
                num_decks=num_decks,
            )

            if n_processes is not None:

                n_processes = mp.cpu_count() if n_processes == -1 else n_processes
                n_batch = n_sims // n_processes + 1

                with mp.Pool(n_processes) as pool:
                    res = pool.map(simulate_games_profit_partial, [n_batch] * n_processes)

                profit = sum(res)
            else:
                profit = simulate_games_profit_partial(n_sims=n_sims)

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
        n_sims: int = 10_000,
        quiet: bool = True,
        basic_strategy: dict = None,
        expected_profit: dict = None,

        double_after_split: bool = True,
        hit_on_soft_17: bool = True,
        num_decks: bool = 6,
):

    decisions = [GameDecision.STAND, GameDecision.HIT, GameDecision.DOUBLE]
    if isinstance(players, str) and len(players) == 2:
        decisions.append(GameDecision.SPLIT)

    decision_profit = {}
    for decision in decisions:
        print('*'*30 + decision + '*'*30)

        decision_profit[decision] = simulate_games_profit(
            n_sims=n_sims,
            player_hand=players,
            dealer_hand=dealer,
            decision=decision,
            quiet=quiet,
            basic_strategy=basic_strategy,
            expected_profit=expected_profit,

            double_after_split=double_after_split,
            hit_on_soft_17=hit_on_soft_17,
            num_decks=num_decks,
        )

        print('Expected Profit:', decision, decision_profit[decision] / n_sims)


def main(
    basic_strategy: str = None,
    expected_value: str = None,
    samples: int = None,
    processes: int = None,
    generate: bool = False,
    double_after_split: bool = True,
    hit_on_soft_17: bool = True,
    num_decks: bool = 6,
):
    if basic_strategy and expected_value:
        df_bs = pd.read_csv(basic_strategy, index_col=0)
        df_ev = pd.read_csv(expected_value, index_col=0)
        json_bs = json.loads(df_bs.to_json())
        json_ev = json.loads(df_ev.to_json())

        bs = {
            (int(player) if player.isnumeric() else player, int(dealer)): decision
            for dealer, player_decision in json_bs.items()
            for player, decision in player_decision.items()
        }

        ev = {
            (int(player) if player.isnumeric() else player, int(dealer)): decision
            for dealer, player_decision in json_ev.items()
            for player, decision in player_decision.items()
        }

        print(df_bs)
        print(df_ev)
        simulate_hand(
            players=11,
            dealer=11,
            n_sims=1000,
            quiet=False,
            basic_strategy=bs,
            expected_profit=ev,

            double_after_split=double_after_split,
            hit_on_soft_17=hit_on_soft_17,
            num_decks=num_decks,
        )

        simulate_hand(
            players=11,
            dealer=11,
            n_sims=100000,
            quiet=True,
            basic_strategy=bs,
            expected_profit=ev,

            double_after_split=double_after_split,
            hit_on_soft_17=hit_on_soft_17,
            num_decks=num_decks,
        )

    if samples:

        df_decision, df_profit = get_basic_strategy(
            n_sims=samples,
            n_processes=processes,
            double_after_split=double_after_split,
            hit_on_soft_17=hit_on_soft_17,
            num_decks=num_decks,
        )

        if generate:
            df_decision.to_csv(os.path.join(DATA_PATH, 'basic_strategy.csv'))
            df_profit.to_csv(os.path.join(DATA_PATH, 'basic_strategy_profit.csv'))

        pprint(df_decision)
        pprint(df_profit)
        print(df_profit.mean().mean())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Simulate Blackjack games and get optimal basic strategy")

    parser.add_argument("-s", "--samples", type=int, default=None,
                        help="Number of samples per player/dealer hand combination [def: 10_000]")
    parser.add_argument("-p", "--processes", type=int, default=None,
                        help="Number of processes to use (-1 for all). "
                             "NOTE: not beneficial for < 10_000 n_sims [def: no parallel]")
    parser.add_argument("-bs", "--basic_strategy", type=str,
                        help="Path to CSV containing basic strategy to load")
    parser.add_argument("-ev", "--expected_value", type=str,
                        help="Path to CSV containing expected_value to load")
    parser.add_argument("-g", "--generate", action='store_true',
                        help="Generate basic strategy and save")
    parser.add_argument("-das", "--double_after_split", action='store_true',
                        help="Allow double after splits")
    parser.add_argument("-H17", "--hit_on_soft_17", action='store_true',
                        help="Dealer hits on soft 17")
    parser.add_argument("-nd", "--num_decks", type=int,
                        help="Number of decks in the shoe", default=6)

    args = parser.parse_args()

    main(
        basic_strategy=args.basic_strategy,
        expected_value=args.expected_value,
        samples=args.samples,
        processes=args.processes,
        generate=args.generate,
        double_after_split=args.double_after_split,
        hit_on_soft_17=args.hit_on_soft_17,
        num_decks=args.num_decks,
    )
