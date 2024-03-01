from blackjack.utils import GameDecision
from blackjack.game import Blackjack
import matplotlib.pyplot as plt
import pandas as pd
import json


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def martingale(
    basic_strategy: dict = None,
    double_after_split: bool = True,
    hit_on_soft_17: bool = True,
    num_decks: bool = 6,

    bank: int = 100000,
    initial_bet: int = 10,
):

    initial_bank = bank
    bank_hist = [bank]

    n_consecutive_losses = 0
    n_games = 0
    while bank > 0:
        game = Blackjack(
            n_packs=num_decks,
            double_after_split=double_after_split,
            hit_on_soft_17=hit_on_soft_17,
        )

        dealer_hand = game.dealer.hand.cards[0].value
        dealer_hand = dealer_hand if isinstance(dealer_hand, int) else 11

        player = game.players[0]
        stake_amount = min(2 ** n_consecutive_losses * initial_bet, bank)
        player.stake = stake_amount  # TODO: remove monkey patching!!

        while not game.is_dealer_turn:
            player = game.next_player()

            if player.hand.is_splittable:
                player_hand = str(player.hand).replace('J', 'T').replace('Q', 'T').replace('K', 'T')

                if player_hand.isnumeric():
                    player_hand = int(player_hand)

                if player_hand == 'AA':
                    player_hand = 12

            elif player.hand.is_soft_value:
                player_hand = player.hand.get_string_rep()
            else:
                player_hand = player.hand.value

            opt_decision = basic_strategy[(player_hand, dealer_hand)]

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

        profit = game.get_players_profit()[0]
        bank += profit
        bank = max(0, bank)  # technically not the right way to do this
        bank_hist.append(bank)

        n_games += 1
        n_consecutive_losses = 0 if profit > 0 else n_consecutive_losses + 1

    print(bank_hist)
    plt.plot(bank_hist)
    plt.title('Martingale Strategy')
    plt.suptitle(f'Bank: €{initial_bank:,}. Stake: €{initial_bet:,}')
    plt.grid()
    plt.show()


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
                        help="Number of decks in the shoe", default=6)

    args = parser.parse_args()

    df_bs = pd.read_csv(args.basic_strategy, index_col=0)
    print(df_bs)
    json_bs = json.loads(df_bs.to_json())

    bs = {
        (int(player) if player.isnumeric() else player, int(dealer)): decision
        for dealer, player_decision in json_bs.items()
        for player, decision in player_decision.items()
    }

    martingale(
        basic_strategy=bs,
        double_after_split=args.double_after_split,
        hit_on_soft_17=args.hit_on_soft_17,
        num_decks=args.num_decks,
    )
