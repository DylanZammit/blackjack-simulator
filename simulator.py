from typing import Union
from random import randint
from pprint import pprint
import pandas as pd
from itertools import product, cycle
from dataclasses import dataclass


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


def is_splittable(hand):
    return len(set(hand)) == 1 and len(hand) == 2


def is_soft_value(hand):
    return 'A' in hand and len(hand) == 2


@dataclass
class GameDecision:
    SPLIT: str = 'SP'
    STAND: str = 'S'
    HIT: str = 'H'


@dataclass
class GameState:
    WON: str = 'WON'
    LOST: str = 'LOST'
    DRAW: str = 'DRAW'
    LIVE: str = 'LIVE'
    STAND: str = 'STAND'


# class Hand:
#     def __init__(self, hand: str = ''):
#         self.hand = hand
#
#     def __len__(self):
#         return len(self.hand)
#
#     def __repr__(self):
#         return self.hand

class Player:
    def __init__(
            self,
            name: Union[str, int],
            hand: str = '',
    ):
        self.name = name
        self.hand = hand
        self.status = GameState.LIVE
        self.decision_hist = []

    def deal_card(self, card: str):
        self.hand += card

    def log_decision(self, decision):
        self.decision_hist.append((self.hand, decision))

    @property
    def card_count(self):
        return len(self.hand)

    @property
    def hand_value(self):
        return hand2val(self.hand)

    @property
    def is_blackjack(self):
        return self.hand_value == 21 and self.card_count == 2

    @property
    def is_finished(self):
        return self.status in ['WON', 'LOST', 'DRAW']

    @property
    def is_splittable(self):
        return len(set(self.hand)) == 1 and self.card_count == 2

    @property
    def is_soft_value(self):
        return 'A' in self.hand and self.card_count == 2  # ?


class Blackjack:

    def __init__(
            self,
            n_packs: int = None,
            n_players: int = 1
    ):
        cards = 'A23456789TJQK'
        self.n_packs = n_packs
        self.shoe = list(cards) * n_packs if n_packs is not None else list(cards)
        self.dealer = Player('dealer')
        self.players = [Player(i) for i in range(n_players)]

        self.draw = self.draw_card()
        self.__setup()

        self.player_turn = 0

    def next_player(self):

        if all([player.is_finished or player.status == GameState.STAND for player in self.players]):
            self.hit_dealer()
            return None

        is_finished = True
        player = None
        while is_finished:
            self.player_turn += 1
            if self.player_turn == self.n_players:
                self.player_turn = 0
            player = self.players[self.player_turn]
            is_finished = player.is_finished

        return player

    def draw_card(self):
        while not self.is_finished:
            if self.n_packs is not None:
                yield self.shoe.pop(randint(1, len(self.shoe) - 1))
            else:
                yield self.shoe[randint(1, len(self.shoe) - 1)]
        yield -1

    def __setup(self):

        for player in self.players:
            player.deal_card(next(self.draw))
            player.deal_card(next(self.draw))

        self.dealer.deal_card(next(self.draw))
        self.dealer.deal_card(next(self.draw))

        for player in self.players:
            if player.hand_value == self.dealer.hand_value == 21:
                player.status = GameState.DRAW
            elif self.dealer.hand_value == 21:
                player.status = GameState.LOST

    def hit_dealer(self):

        while self.dealer.hand_value <= 16:
            self.dealer.deal_card(next(self.draw))

        if self.dealer.hand_value > 21:
            for player in self.players:
                if player.is_finished: continue
                player.status = GameState.WON
            return

        for player in self.players:
            if player.is_finished: continue
            if self.dealer.hand_value == player.hand_value:
                if player.is_blackjack == self.dealer.is_blackjack:
                    player.status = GameState.DRAW

                if player.is_blackjack:
                    player.status = GameState.WON

                if self.dealer.is_blackjack:
                    player.status = GameState.LOST
                return

            player.status = GameState.WON if self.dealer.hand_value < player.hand_value else GameState.LOST

    def hit(self):
        return self.__hit(self.next_player())

    def __hit(self, player=None):
        if player is None: return

        if player.hand_value == 21: return
        if player.card_count > 1:  # can only happen on split
            player.log_decision(GameDecision.HIT)
        player.deal_card(next(self.draw))

        if player.hand_value > 21:
            player.status = GameState.LOST
        return GameDecision.HIT

    def stand(self):
        return self.__stand(self.next_player())

    def __stand(self, player):
        player.log_decision(GameDecision.STAND)
        player.status = GameState.STAND
        return GameDecision.STAND

    def split(self):
        return self.__split(self.next_player())

    def __split(self, player):
        if player is None: return
        if player.is_splittable:

            player.log_decision(GameDecision.SPLIT)
            player.hand = player.hand[0]
            new_player = Player(name=player.name, hand=player.hand)

            self.players.insert(self.player_turn+1, new_player)
            self.__hit(player=player)
            self.__hit(player=new_player)
            self.player_turn += 1

            return GameDecision.SPLIT

        self.__hit(player=player)
        return GameDecision.HIT

    def play_random(self):
        player = self.next_player()
        if player is None: return
        decision = randint(1, 3) if player.is_splittable else randint(1, 2)

        if decision == 1:
            return self.__hit(player)
        elif decision == 2:
            return self.__stand(player)
        elif decision == 3:
            return self.__split(player)

    def play_full_random(self):
        while not self.is_finished:
            self.play_random()

    @property
    def is_finished(self):
        return all(player.is_finished for player in self.players)

    @property
    def n_players(self):
        return len(self.players)

    def __repr__(self):
        out = ['*' * 30, f'Dealer Hand: {self.dealer.hand} ({self.dealer.hand_value})']

        for player in self.players:
            out.append(f'({player.status}) Player {player.name} Hand: {player.hand} ({player.hand_value})')

        return '\n'.join(out)


def get_win_ratio(x):
    if x[GameState.WON] + x[GameState.LOST] == 0: return 0.5
    return x[GameState.WON] / (x[GameState.WON] + x[GameState.LOST])


def get_basic_strategy(n_sims=100_000):

    def_state = {GameState.WON: 0, GameState.LOST: 0, GameState.DRAW: 0}
    cards = 'A23456789T'
    player_starting_vals = list(range(5, 22)) + [f'{c},{c}' for c in cards] + [f'A,{c}' for c in cards[1:-1]]
    dealer_starting_vals = range(2, 12)
    outcomes = {
        k: {GameDecision.HIT: def_state.copy(), GameDecision.STAND: def_state.copy(), GameDecision.SPLIT: def_state.copy()}
        for k in product(player_starting_vals, dealer_starting_vals)
    }
    for i in range(1, n_sims + 1):
        if i % 5_000 == 0: print(f'{int(i/n_sims*100)}%')
        game = Blackjack(n_players=10)
        game.play_full_random()

        dealer_first_card = card2val(game.dealer.hand[0])

        for player in game.players:
            for hand, decision in player.decision_hist:
                hand = hand.replace('J', 'T').replace('Q', 'T').replace('K', 'T')
                if is_splittable(hand):
                    outcomes[(f'{hand[0]},{hand[1]}', dealer_first_card)][decision][player.status] += 1
                elif is_soft_value(hand):
                    c = hand[0] if hand[1] == 'A' else hand[1]
                    outcomes[(f'A,{c}', dealer_first_card)][decision][player.status] += 1
                else:
                    outcomes[(hand2val(hand), dealer_first_card)][decision][player.status] += 1

    pprint(outcomes)

    best_play = {hands: GameDecision.HIT if get_win_ratio(outcomes[hands][GameDecision.HIT]) > get_win_ratio(outcomes[hands][GameDecision.STAND]) else GameDecision.STAND
                 for hands, res in outcomes.items()}
    df_best_play = pd.DataFrame(best_play.values(), index=list(best_play.keys()))
    df_best_play.index = pd.MultiIndex.from_tuples(df_best_play.index, names=['player', 'dealer'])
    df_best_play = df_best_play.unstack().replace({1: GameDecision.HIT, 0: GameDecision.STAND})
    return df_best_play


if __name__ == '__main__':
    df = get_basic_strategy()
    pprint(df)
