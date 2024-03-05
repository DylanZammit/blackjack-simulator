from blackjack.game import Blackjack, Player, Hand, GameDecision
from blackjack.hand import format_hand


class Simulation:
    def __init__(
            self,
            rounds_per_game: int,
            initial_bet: int,
            initial_bank: int,
            n_packs: int,
            double_after_split: bool,
            hit_on_soft_17: bool,
            basic_strategy: dict = None,
            **kwargs
    ):
        self.initial_bet = initial_bet
        self.bank = initial_bank

        self.initial_bank = initial_bank
        self.n_packs = n_packs
        self.double_after_split = double_after_split
        self.hit_on_soft_17 = hit_on_soft_17
        self.kwargs = kwargs
        self.rounds_per_game = rounds_per_game

        self.n_consecutive_losses = 0

        self.player = Player(stake=self.initial_bet)

        self.game = Blackjack(
            players=[self.player],
            n_packs=self.n_packs,
            double_after_split=self.double_after_split,
            hit_on_soft_17=self.hit_on_soft_17,
        )

        self.basic_strategy = basic_strategy

        self.bank_hist = []
        self.true_count_hist = []

        self.total_stake = 0
        self.total_profit = 0

    @property
    def house_edge(self):
        edge = - self.total_profit / self.total_stake
        return edge

    def strategy(self, player_hand: Hand, dealer_hand: int) -> GameDecision:
        if self.basic_strategy is None:
            err_msg = 'Must provide a basic strategy, otherwise override this method with a customer strategy'
            raise NotImplementedError(err_msg)

        player_hand_fmt = format_hand(player_hand)

        opt_decision = GameDecision(self.basic_strategy[(player_hand_fmt, dealer_hand)])

        # Can only double on first two cards
        # Using known optimal decisions if double not allowed
        if opt_decision.value == GameDecision.DOUBLE and len(player_hand) != 2:
            # TODO: Cater for this case in basic strategy generation
            if not player_hand.is_soft_value or player_hand.value() <= 17:
                opt_decision = GameDecision.HIT
            else:
                opt_decision = GameDecision.STAND
        return opt_decision

    def get_stake(self) -> int:
        raise NotImplementedError('Must set up a stake strategy')

    def run(self):
        game = self.game
        for _ in range(self.rounds_per_game):

            dealer_hand = game.dealer.hand.cards[0].value
            dealer_hand = dealer_hand if isinstance(dealer_hand, int) else 11

            while not game.is_dealer_turn:
                player_hand = game.next_hand()
                opt_decision = self.strategy(player_hand, dealer_hand)
                getattr(game, opt_decision.value)()

            game.hit_dealer()

            self.total_stake += self.player.stake
            self.total_profit += self.player.profit

            # technically not the right way to do this
            self.bank = max(0, self.bank + self.player.profit)

            self.bank_hist.append(self.bank)
            self.true_count_hist.append(game.true_count)

            self.n_consecutive_losses = 0 if self.player.profit > 0 else (self.n_consecutive_losses + 1)
            stake = self.get_stake()

            self.player = Player(stake=stake)
            game.new_game(players=[self.player])
        return self
