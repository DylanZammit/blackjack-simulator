from blackjack.simulator import Simulation
import matplotlib.pyplot as plt
import pandas as pd
from blackjack.utils import csv_to_dict

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class BasicStrategy(Simulation):

    def get_stake(self) -> int:
        return self.initial_bet


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Simulate Blackjack games and get optimal basic strategy")

    parser.add_argument("-bs", "--basic_strategy", type=str,
                        help="Path to CSV containing basic strategy to load", required=True)
    parser.add_argument("-das", "--double_after_split", action='store_true',
                        help="Allow double after splits")
    parser.add_argument("-H17", "--hit_on_soft_17", action='store_true',
                        help="Dealer hits on soft 17")
    parser.add_argument("-nd", "--num_decks", type=int,
                        help="Number of decks in the shoe [def=6]", default=6)
    parser.add_argument("-br", "--bankroll", type=int,
                        help="Bank amount of player [def=1000]", default=1000)
    parser.add_argument("-s", "--stake", type=int,
                        help="Initial stake [def=10]", default=10)
    parser.add_argument("-rpg", "--rounds_per_game", type=int,
                        help="Rounds per game to play [def=1000]", default=1000)
    parser.add_argument("-n", "--num_games", type=int,
                        help="Number of games to play [def=100]", default=100)
    # parser.add_argument("-p", "--plot", action='store_true',
    #                     help="Plot Chart")

    args = parser.parse_args()

    bs = csv_to_dict(args.basic_strategy)

    initial_bank = args.bankroll
    initial_bet = args.stake
    bank_hists = []

    house_edge = 0

    ax = None
    for i in range(args.num_games):
        print(f'Sim {i}')
        sim = BasicStrategy(
            rounds_per_game=args.rounds_per_game,
            basic_strategy=bs,
            double_after_split=args.double_after_split,
            hit_on_soft_17=args.hit_on_soft_17,
            n_packs=args.num_decks,
            initial_bank=initial_bank,
            initial_bet=initial_bet,
        ).run()

        bank_hists.append(sim.bank_hist)

        house_edge += sim.house_edge

    player_edge = - house_edge / args.num_games * 100

    title = f'Basic Strategy (Player Edge: {player_edge:.2f}%)'
    print(title)

    # if args.plot:
    for bh in bank_hists:
        ax = plt.plot(bh, color='black', alpha=0.2)

    df_hists = pd.DataFrame(bank_hists)
    df_hists.mean().plot(color='red', alpha=1)

    plt.title(title)
    plt.suptitle(f'Bank: €{initial_bank:,}. Stake: €{initial_bet:,}')
    plt.grid()
    plt.show()

