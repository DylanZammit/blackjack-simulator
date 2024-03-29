from blackjack.utils import GameDecision, GameState
from blackjack.hand import Hand
from blackjack.card import Card
from blackjack.player import Player
from typing import List, Generator
from random import shuffle
from math import ceil


class Blackjack:
    CARDS = 'A23456789TJQK'

    def __init__(
            self,
            n_packs: int = 4,
            players: List[Player] = None,
            n_players: int = 1,
            dealer_hand: Hand = None,
            hit_on_soft_17: bool = True,
            double_after_split: bool = False,
            quiet: bool = True,
    ):
        self.n_packs = n_packs
        self.shoe = []
        self.discarded = list(self.CARDS) * self.n_packs * 4

        self.hit_on_17 = hit_on_soft_17
        self.double_after_split = double_after_split

        self.draw = self.draw_card()
        self.quiet = quiet

        self.dealer = Player('dealer', hand=dealer_hand)

        self.players = [Player(i) for i in range(n_players)] if not players else players
        self.is_finished = False

        self.__setup()

        self.current_player = self.players[0]

    def __setup(self) -> None:

        # set 0.2 as parameter or cut card
        if self.is_time_to_shuffle:
            self.shuffle()

        if self.is_finished:
            self.dealer = Player('dealer')

        # remove cards from shoe
        for player in self.players + [self.dealer]:
            for hand in player.hands:
                for card in hand.cards:
                    self.discarded.append(str(card))
                    self.shoe.remove(str(card))

        for player in self.players + [self.dealer]:
            for hand in player.hands:
                remaining_cards = 2 - len(hand)
                for _ in range(remaining_cards):
                    hand.deal_card(next(self.draw))

        self.player_name_map = {p.name: p for p in self.players}

        self.is_finished = False
        if self.dealer.hand.is_blackjack:
            while self.next_hand() is not None:
                self.stand()
            self.hit_dealer()
            self.is_finished = True

    @property
    def is_time_to_shuffle(self):
        return len(self.shoe) / (len(self.shoe) + len(self.discarded)) < 0.25

    def shuffle(self):
        self.shoe = list(self.CARDS) * self.n_packs * 4
        shuffle(self.shoe)
        self.discarded = []

    def new_game(self, players: List[Player] = None):
        self.players = [Player(i) for i in range(self.n_players)] if not players else players
        self.__setup()

    @property
    def high_low_count(self):
        return sum(Card(c).high_low_count for c in self.discarded)

    @property
    def true_count(self):
        return ceil(self.high_low_count / self.n_packs)

    @property
    def game_round(self):
        if self.is_dealer_turn:
            return -1
        return min(hand.round for player in self.players for hand in player.hands if hand.status.value == GameState.LIVE) + 1

    def next_hand(self) -> Hand | None:
        for player in self.players:
            for hand in player.hands:
                if hand.round < self.game_round and hand.status.value == GameState.LIVE:
                    return hand

    def hands(self) -> Generator[Hand, None, None]:
        for player in self.players:
            for hand in player.hands:
                yield hand

    def draw_card(self):
        while len(self.shoe) > 0:
            c = self.shoe.pop()
            self.discarded.append(c)
            yield Card(c)
        yield -1

    def hit_dealer(self) -> None:
        dealer_hand = self.dealer.hand
        if not self.is_dealer_turn:
            if not self.quiet:
                print('It is not the dealer\'s turn!')
            return

        if self.is_finished:
            if not self.quiet:
                print('Game is already finished!')
            return

        while dealer_hand.value() <= 16 or \
                dealer_hand.value() == 17 and dealer_hand.is_soft_value and self.hit_on_17:
            dealer_hand.deal_card(next(self.draw))

        for hand in self.hands():
            if hand.is_finished:
                continue
            if dealer_hand.value() > 21:
                hand.status = GameState.WON
            elif dealer_hand < hand:
                hand.status = GameState.WON
            elif dealer_hand > hand:
                hand.status = GameState.LOST
            elif dealer_hand == hand:
                hand.status = GameState.DRAW

        self.is_finished = True

    def double(self):
        hand = self.next_hand()
        if hand is None:
            return

        assert hand.value != 21, f'Cannot double on {hand}!'
        assert len(hand) == 2, f'Can only double on starting hands, current hand: {hand}'
        assert self.double_after_split or not hand.is_split, 'Cannot double after split!'

        hand.stake *= 2
        hand.is_doubled = True
        hand.log_decision(GameDecision.DOUBLE)
        hand.deal_card(next(self.draw))
        hand.status = GameState.LOST if hand.value() > 21 else GameState.STAND
        hand.round += 1

    def hit(self) -> None:
        hand = self.next_hand()
        if hand is None:
            return

        if hand.is_finished:
            return

        assert hand.value() != 21, f'Cannot hit on {hand}!'

        hand.log_decision(GameDecision.HIT)

        card = next(self.draw)
        hand.deal_card(card)

        if hand.value() > 21:
            hand.status = GameState.LOST
        hand.round += 1

    def stand(self) -> None:
        hand = self.next_hand()
        if hand is None:
            return

        hand.log_decision(GameDecision.STAND)
        hand.status = GameState.STAND
        hand.round += 1

    def split(self) -> None:
        hand = self.next_hand()
        if hand is None:
            return

        hand.log_decision(GameDecision.SPLIT)

        assert hand.is_splittable, f'Cannot Split {hand}!'

        old_hand = Hand([hand.cards[0]], is_split=True, player_name=hand.player_name, stake=hand.stake)
        new_hand = Hand([hand.cards[0]], is_split=True, player_name=hand.player_name, stake=hand.stake)
        old_hand.round = self.game_round
        new_hand.round = self.game_round

        player = self.player_name_map[hand.player_name]

        hand_idx = player.hands.index(hand)
        player.hands.remove(hand)

        old_hand.deal_card(next(self.draw))
        new_hand.deal_card(next(self.draw))

        player.hands.insert(hand_idx, new_hand)
        player.hands.insert(hand_idx, old_hand)
        hand.round += 1

    @property
    def is_dealer_turn(self) -> bool:
        return all(hand.is_idle or hand.is_finished for hand in self.hands())

    @property
    def n_players(self) -> int:
        return len(self.players)

    def __repr__(self):
        out = ['*' * 30, f'Dealer Hand: {self.dealer.hand.cards} ({self.dealer.hand.value()})']

        for player in self.players:
            for hand in player.hands:
                out.append(f'({hand.status}) Player {player.name} Hand: {hand.cards} ({hand.value()})...........{hand.decision_hist}')

        return '\n'.join(out)
