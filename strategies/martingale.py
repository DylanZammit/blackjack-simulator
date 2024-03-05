from blackjack.simulator import Simulation
from blackjack.utils import csv_to_dict
import matplotlib.pyplot as plt
import pandas as pd


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class Martingale(Simulation):

    def get_stake(self) -> int:
        return min(2 ** self.n_consecutive_losses * self.initial_bet, self.bank)


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

    args = parser.parse_args()

    bs = csv_to_dict(args.basic_strategy)

    initial_bank = args.bankroll
    initial_bet = args.stake
    bank_hists = []

    ax = None
    for i in range(args.num_games):
        print(f'Sim {i}')
        sim = Martingale(
            rounds_per_game=args.rounds_per_game,
            basic_strategy=bs,
            double_after_split=args.double_after_split,
            hit_on_soft_17=args.hit_on_soft_17,
            n_packs=args.num_decks,
            initial_bank=initial_bank,
            initial_bet=initial_bet,
        ).run()

        ax = plt.plot(sim.bank_hist, color='black', alpha=0.2)
        bank_hists.append(sim.bank_hist)

    df_hists = pd.DataFrame(bank_hists)
    df_hists.mean().plot(color='red', alpha=1)

    plt.title('Martingale Strategy')
    plt.suptitle(f'Bank: €{initial_bank:,}. Stake: €{initial_bet:,}')
    plt.grid()
    plt.show()

