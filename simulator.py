from typing import Union
from random import randint


def card2val(card: str) -> Union[int, tuple[int, int]]:
    if card == 'A': return 1, 11
    if card in 'TJQK': return 10
    return int(card)


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
            n_packs: int = 3,
    ):
        cards = 'A23456789TJQK'
        self.n_packs = n_packs
        self.shoe = list(cards) * n_packs
        self.dealer = ''
        self.player = ''
        self.status = 'LIVE'

        self.draw = self._draw()

        self.__setup()

    def _draw(self):
        while self.status == 'LIVE':
            yield self.shoe.pop(randint(1, len(self.shoe) - 1))
        yield -1

    def __setup(self):
        self.player += next(self.draw)
        self.player += next(self.draw)

        self.dealer += next(self.draw)
        self.dealer += next(self.draw)  # do I need to draw here?

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
            '*' * 10 + self.status + '*' * 10,
            f'Dealer Hand: {self.dealer} ({self.dealer_val})',
            f'Player Hand: {self.player} ({self.player_val})',
        ])


if __name__ == '__main__':
    n_sims = 10_000

    won, draw, lost = 0, 0, 0

    for _ in range(n_sims):
        game = Blackjack()
        if game.player_val <= 15:
            game.deal()
        game.stand()

        if game.status in ['WON', 'BLACKJACK']:
            won += 1
        elif game.status == 'DRAW':
            draw += 1
        elif game.status == 'LOST':
            lost += 1
        print(game)

    print(f'{lost=} {draw=} {won=}')
    print(f'lost={int(lost/n_sims*100)}% draw={int(draw/n_sims*100)}% won={int(won/n_sims*100)}%')
