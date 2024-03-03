from blackjack.utils import GameDecision, GameState
from blackjack.hand import Hand
from blackjack.card import Card
from blackjack.player import Player
from typing import List, Generator
from random import shuffle


class Blackjack:

    def __init__(
            self,
            n_packs: int = 4,
            n_players: int = 1,
            player_hands: List[Hand] = None,
            dealer_hand: Hand = None,
            hit_on_soft_17: bool = True,
            double_after_split: bool = False,
            quiet: bool = True,
    ):
        cards = 'A23456789TJQK'
        self.hit_on_17 = hit_on_soft_17
        self.n_packs = n_packs
        self.shoe = list(cards) * n_packs * 4
        self.double_after_split = double_after_split
        shuffle(self.shoe)

        self.draw = self.draw_card()
        self.quiet = quiet

        self.dealer = Player('dealer', hand=dealer_hand)

        for card in self.dealer.hand.cards:
            self.shoe.remove(str(card))

        if player_hands is not None:
            self.players = [Player(i, hand=hand) for i, hand in enumerate(player_hands)]
            assert len(player_hands) == n_players, 'There must be predefined hands as there are players'
            assert dealer_hand is not None, 'Must also specify the dealer\'s hand if the players\' hand is specified'

            # remove cards from shoe
            for hand in player_hands:
                for card in hand.cards:
                    self.shoe.remove(str(card))

            if len(self.dealer.hand) == 1:
                self.dealer.hand.deal_card(next(self.draw))

            self.is_finished = False
            if self.dealer.hand.is_blackjack:
                while self.next_hand() is not None:
                    self.stand()
                self.hit_dealer()
                self.is_finished = True

        else:
            self.players = [Player(i) for i in range(n_players)]
            self.new_game()

        self.player_name_map = {p.name: p for p in self.players}
        self.player_turn = 0
        self.current_player = self.players[0]

    def new_game(self):
        for player in self.players:
            player.reset()

        self.__setup()
        self.is_finished = False
        if self.dealer.hand.is_blackjack:
            while self.next_hand() is not None:
                self.stand()
            self.hit_dealer()
            self.is_finished = True

    @property
    def game_round(self):
        if self.is_dealer_turn:
            return -1
        return min(hand.round for player in self.players for hand in player.hands if hand.status.value == GameState.LIVE) + 1

    def next_hand(self) -> Hand | None:
        if self.dealer.hand.is_blackjack:
            return

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
            yield Card(self.shoe.pop())
        yield -1

    def __setup(self) -> None:

        for _ in range(2):
            for player in self.players:
                for hand in player.hands:
                    hand.deal_card(next(self.draw))

        self.dealer.hand.deal_card(next(self.draw))

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

        old_hand = Hand([hand.cards[0]], is_split=True, player_name=hand.player_name)
        new_hand = Hand([hand.cards[0]], is_split=True, player_name=hand.player_name)
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
                out.append(f'({hand.status}) Player {player.name} Hand: {hand.cards} ({hand.value()})')

        return '\n'.join(out)
