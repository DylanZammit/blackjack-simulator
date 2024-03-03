from blackjack.utils import GameDecision
from blackjack.game import Blackjack
from blackjack.player import Player
from blackjack.hand import format_hand
import matplotlib.pyplot as plt
import pandas as pd
import json


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def play_basic_strategy(
    basic_strategy: dict = None,
    double_after_split: bool = True,
    hit_on_soft_17: bool = True,
    num_decks: bool = 6,

    bank: int = 100000,
    initial_bet: int = 10,
):

    initial_bank = bank
    bank_hist = [bank]

    n_games = 0
    while bank > 0:
        player = Player(stake=initial_bet)
        game = Blackjack(
            players=[player],
            n_packs=num_decks,
            double_after_split=double_after_split,
            hit_on_soft_17=hit_on_soft_17,
        )

        dealer_hand = game.dealer.hand.cards[0].value
        dealer_hand = dealer_hand if isinstance(dealer_hand, int) else 11

        while not game.is_dealer_turn:
            hand = game.next_hand()
            player_hand = format_hand(hand)

            opt_decision = basic_strategy[(player_hand, dealer_hand)]

            # Can only double on first two cards. Hit otherwise
            if opt_decision == GameDecision.DOUBLE and len(hand) != 2:
                opt_decision = GameDecision.HIT.value

            getattr(game, opt_decision)()

        game.hit_dealer()

        bank += player.profit
        bank = max(0, bank)  # technically not the right way to do this
        bank_hist.append(bank)

        n_games += 1

    print(bank_hist)
    plt.plot(bank_hist)
    plt.title('Basic Strategy')
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

    play_basic_strategy(
        basic_strategy=bs,
        double_after_split=args.double_after_split,
        hit_on_soft_17=args.hit_on_soft_17,
        num_decks=args.num_decks,
        bank=1000,
        initial_bet=10,
    )
