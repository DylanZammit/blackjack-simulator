from blackjack.Simulator import Simulation
import matplotlib.pyplot as plt
import pandas as pd
import json


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class HighLow(Simulation):

    def get_stake(self) -> int:
        betting_unit = self.bank // 100
        multiplier = 0 if self.game.is_time_to_shuffle else min(4, (self.game.true_count - 1))
        stake = max(self.initial_bet, multiplier * betting_unit)
        return stake


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
                        help="Bank amount of player [def=10000]", default=100_000)
    parser.add_argument("-s", "--stake", type=int,
                        help="Initial stake [def=10]", default=1)
    parser.add_argument("-rpg", "--rounds_per_game", type=int,
                        help="Rounds per game to play [def=1000]", default=1000)
    parser.add_argument("-n", "--num_games", type=int,
                        help="Number of games to play [def=100]", default=100)
    # parser.add_argument("-p", "--plot", action='store_true',
    #                     help="Plot Chart")

    args = parser.parse_args()

    df_bs = pd.read_csv(args.basic_strategy, index_col=0)
    print(df_bs)
    json_bs = json.loads(df_bs.to_json())

    bs = {
        (int(player) if player.isnumeric() else player, int(dealer)): decision
        for dealer, player_decision in json_bs.items()
        for player, decision in player_decision.items()
    }

    initial_bank = args.bankroll
    initial_bet = args.stake
    betting_unit = initial_bank // 100

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].set_title(f'Bank: €{initial_bank:,}. Betting Unit: €{betting_unit:,}')
    ax[0].grid()

    ax[1].set_title('True Count')
    ax[1].grid()

    bank_hists = []
    true_count_hists = []
    house_edge = 0

    for i in range(args.num_games):
        print(f'Sim {i}')
        sim = HighLow(
            rounds_per_game=args.rounds_per_game,
            basic_strategy=bs,
            double_after_split=args.double_after_split,
            hit_on_soft_17=args.hit_on_soft_17,
            n_packs=args.num_decks,
            initial_bank=initial_bank,
            initial_bet=initial_bet,
        ).run()

        bank_hists.append(sim.bank_hist)
        true_count_hists.append(sim.true_count_hist)
        house_edge += sim.house_edge

    house_edge = house_edge / args.num_games * 100
    title = f'High-Low Strategy: (House Edge: {house_edge:.2f}%)'

    # if args.plot:
    for bh, tch in zip(bank_hists, true_count_hists):
        ax[0].plot(bh, color='black', alpha=0.2)
        ax[1].plot(tch, color='black', alpha=0.1)

    fig.suptitle(title)

    df_hists = pd.DataFrame(bank_hists)
    df_hists.mean().plot(ax=ax[0], color='red', alpha=1, grid=True)

    plt.show()
