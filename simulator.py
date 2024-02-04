from typing import Union
from random import randint, random
from pprint import pprint
import pandas as pd
from itertools import product


def card2val(card: str) -> Union[int, tuple[int, int]]:
    if card == 'A': return 1, 11
    if card in 'TJQK': return 10
    return int(card)


# TODO: optimise
def hand2val(hand: str) -> int:
    num_aces = hand.count('A')
    hard_cards = hand.replace('A', '')
    hard_val = sum(card2val(card) for card in hard_cards)
    soft_val = [hard_val + i + 11 * (num_aces - i) for i in range(num_aces + 1)]
    val = max((sv for sv in soft_val if sv <= 21), default=22)
    return val


class Blackjack:

    WON = 'WON'
    LOST = 'LOST'
    DRAW = 'DRAW'
    LIVE = 'LIVE'

    def __init__(
            self,
            n_packs: int = 4,
            dealer: str = '',
            player: str = '',
    ):
        cards = 'A23456789TJQK'
        self.n_packs = n_packs
        self.shoe = list(cards) * n_packs
        self.dealer = dealer
        self.init_dealer = ''
        self.init_dealer_val = ''
        self.player = player
        self.status = Blackjack.LIVE
        self.is_blackjack = False

        self.draw = self.__hit()
        self.__setup()

        assert len(self.dealer) == 2, f'Dealer must have 2 cards but has {len(self.dealer[0])}'
        assert len(self.player) == 2, f'Player must have 2 cards but has {len(self.player[0])}'

    def __hit(self):
        while self.status == Blackjack.LIVE:
            yield self.shoe.pop(randint(1, len(self.shoe) - 1))
        yield -1

    def __setup(self):

        if len(self.player) == 0:
            self.player += next(self.draw)
            self.player += next(self.draw)
        else:
            pass  # TODO: remove from shoe

        if len(self.dealer) == 0:
            self.dealer += next(self.draw)
            self.dealer += next(self.draw)  # do I need to draw here?
        else:
            pass  # TODO: remove from show

        if self.player_val == self.dealer_val == 21:
            self.status = Blackjack.DRAW
        elif self.dealer_val == 21:
            self.status = Blackjack.LOST

        self.init_dealer = self.dealer[0]
        self.init_dealer_val = hand2val(self.init_dealer)

    def hit(self):
        if self.status != Blackjack.LIVE:
            return

        if self.player_val == 21:
            return

        self.player += next(self.draw)

        if self.player_val > 21:
            self.status = Blackjack.LOST

    def stand(self):
        if self.status != Blackjack.LIVE:
            return

        while self.dealer_val <= 16:
            self.dealer += next(self.draw)

        if self.dealer_val > 21:
            self.status = Blackjack.WON
        elif self.dealer_val == self.player_val:
            if self.is_player_blackjack == self.is_dealer_blackjack:
                self.status = Blackjack.DRAW

            if self.is_player_blackjack:
                self.is_blackjack = True
                self.status = Blackjack.WON

            if self.is_dealer_blackjack:
                self.status = Blackjack.LOST

        elif self.dealer_val < self.player_val:
            self.status = Blackjack.WON
        else:
            self.status = Blackjack.LOST

    @property
    def player_card_count(self):
        return len(self.player)

    @property
    def dealer_card_count(self):
        return len(self.dealer)

    @property
    def is_player_blackjack(self):
        return self.player_val == 21 and self.player_card_count == 2

    @property
    def is_dealer_blackjack(self):
        return self.dealer_val == 21 and self.dealer_card_count == 2

    @property
    def player_val(self):
        return hand2val(self.player)

    @property
    def dealer_val(self):
        return hand2val(self.dealer)

    def __repr__(self):
        return '\n'.join([
            '' * 10 + self.status + '' * 10,
            f'Dealer Hand: {self.dealer} ({self.dealer_val})',
            f'Player Hand: {self.player} ({self.player_val})',
        ])


def get_win_ratio(x):
    if x[Blackjack.WON] + x[Blackjack.LOST] == 0: return 0.5
    return x[Blackjack.WON] / (x[Blackjack.WON] + x[Blackjack.LOST])


def get_basic_strat(n_sims=100_000):

    def_state = {Blackjack.WON: 0, Blackjack.LOST: 0, Blackjack.DRAW: 0}
    player_starting_vals = range(4, 22)
    dealer_starting_vals = range(2, 12)
    outcomes = {k: {'H': def_state.copy(), 'S': def_state.copy()} for k in
                product(player_starting_vals, dealer_starting_vals)}

    for i in range(1, n_sims + 1):
        if i % 5_000 == 0: print(f'{int(i/n_sims*100)}%')
        game = Blackjack()
        states = []
        while game.status == Blackjack.LIVE:
            game_map = (game.player_val, game.init_dealer_val)
            u = randint(0, 1)
            if u == 0 or game.player_val == 21:
                states.append((game_map, 'S'))
                game.stand()
            else:
                states.append((game_map, 'H'))
                game.hit()

        for game_map, decision in states:
            outcomes[game_map][decision][game.status] += 1
    pprint(outcomes)

    best_play = {hands: 'H' if get_win_ratio(outcomes[hands]['H']) > get_win_ratio(outcomes[hands]['S']) else 'S'
                 for hands, res in outcomes.items()}
    df_best_play = pd.DataFrame(best_play.values(), index=list(best_play.keys()))
    df_best_play.index = pd.MultiIndex.from_tuples(df_best_play.index, names=['player', 'dealer'])
    df_best_play = df_best_play.unstack().replace({1: 'H', 0: 'S'})

    return df_best_play


if __name__ == '__main__':
    df = get_basic_strat()
    pprint(df)
