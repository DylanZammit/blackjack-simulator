from typing import Union
from random import randint
from itertools import combinations_with_replacement
from pprint import pprint
import pandas as pd


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

    def __init__(
            self,
            n_packs: int = 4,
            dealer: str = '',
            player: str = ''
    ):
        cards = 'A23456789TJQK'
        self.n_packs = n_packs
        self.shoe = list(cards) * n_packs
        self.dealer = dealer
        self.player = player
        self.status = 'LIVE'

        self.draw = self._draw()
        assert len(self.dealer) in [0, 1], 'Dealer must initially have 0 or 1 cards'
        assert len(self.player) in [0, 2], 'Player must initially have 0 or 2 cards'
        self.__setup()

    def _draw(self):
        while self.status == 'LIVE':
            yield self.shoe.pop(randint(1, len(self.shoe) - 1))
        yield -1

    def __setup(self):

        if len(self.player) == 0:
            self.player += next(self.draw)
            self.player += next(self.draw)
        else:
            pass  # remove from shoe

        if len(self.dealer) == 0:
            self.dealer += next(self.draw)
            self.dealer += next(self.draw)  # do I need to draw here?
        else:
            pass  # remove from show

        if self.player_val == self.dealer_val == 21:
            self.status = 'DRAW'
        elif self.dealer_val == 21:
            self.status = 'LOST'

    def deal(self):
        if self.status != 'LIVE':
            return

        if self.player_val == 21:
            return

        self.player += next(self.draw)

        if self.player_val > 21:
            self.status = 'LOST'

    def stand(self):
        if self.status != 'LIVE':
            return

        while self.dealer_val <= 16:
            self.dealer += next(self.draw)

        if self.dealer_val > 21:
            self.status = 'WON'
        elif self.dealer_val == self.player_val:
            if self.is_player_blackjack == self.is_dealer_blackjack:
                self.status = 'DRAW'

            if self.is_player_blackjack:
                self.status = 'BLACKJACK'

            if self.is_dealer_blackjack:
                self.status = 'LOST'

        elif self.dealer_val < self.player_val:
            self.status = 'WON'
        else:
            self.status = 'LOST'

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


if __name__ == '__main__':
    n_sims = 5_000
    n_packs = 4
    cards = 'A23456789T'

    player_combs = list(combinations_with_replacement(cards, 2))
    combs = []
    for pc in combinations_with_replacement(cards, 2):
        for c in cards:
            combs.append((''.join(pc), c))

    outcomes = {}
    for player, dealer in combs:
        print(player, dealer)
        p_val, d_val = hand2val(player), hand2val(dealer)
        outcomes[(p_val, d_val)] = {}
        for n_hits in range(2):
            outcomes[(p_val, d_val)][n_hits] = {'won': 0, 'draw': 0, 'lost': 0}
            for _ in range(n_sims):
                game = Blackjack(n_packs=n_packs, player=player, dealer=dealer)

                for _ in range(n_hits):
                    game.deal()
                game.stand()

                if game.status in ['WON', 'BLACKJACK']:
                    outcomes[(p_val, d_val)][n_hits]['won'] += 1
                elif game.status == 'DRAW':
                    outcomes[(p_val, d_val)][n_hits]['draw'] += 1
                elif game.status == 'LOST':
                    outcomes[(p_val, d_val)][n_hits]['lost'] += 1

    best_outcome = {}
    for player_dealer, options in outcomes.items():
        n_hits = max(options, key=lambda key: options[key]['won'] / (options[key]['won'] + options[key]['lost']))
        win_pct = options[n_hits]['won'] / (options[n_hits]['won'] + options[n_hits]['lost'])
        best_outcome[player_dealer] = (n_hits, win_pct)
    best_decision = {k: v[0] for k, v in best_outcome.items()}

    df = pd.DataFrame(best_decision.values(), index=list(best_decision.keys()))
    df.index = pd.MultiIndex.from_tuples(df.index, names=['player', 'dealer'])
    df = df.unstack().replace({1: 'H', 0: 'S'})
    pprint(df)
