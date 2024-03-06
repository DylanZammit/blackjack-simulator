# Blackjack Simulator
## Table of contents
## Introduction
In this project I wanted to explore how Blackjack (BJ) is played, the optimal way of playing BJ, and how to beat the odds.

The objective of the game is for your hand to be as close as possible (but not exceed) a total value of 21, where `TJKQ` all count as 10. 
You mean the amount staked if 21 is not exceeded, and the dealer has a value lower than your hand, or has exceeded 21 themselves.
If the first two cards of a hand are an `A` and a `TJQ` or `K`, hence amounting to a 21, that is called a blackjack, which pays out 3 to 2.

More in-depth rules can be found in the [Blackjack Wikipedia page](https://en.wikipedia.org/wiki/Blackjack#Rules_of_play_at_casinos).

## Code
### Installation
Below are the requirements/instructions to be able to deploy and run this code

* `python==3.10`
* Go to the root directory of the project, containing the `setup.py` file and run `pip install .`.
* Run `pip install -r requirements.txt` to install necessary dependencies.

### Object Structure
The `blackjack` package contains various scripts containing classes required for a game of blackjack, following the below logic
* A `game.Blackjack` can contain at least one
* `player.Player`, each of which having at least one
* `hand.Hand`, each of which containing at least two
* `card.Card`, which are of the following possible ranks: `A23456789TJQK`.

The `utils.py` contains common functions and variables, most importantly two `Enum` classes: `utils.GameDecision` and `utils.GameState`.

The possible decisions that can be taken by a player (depending on the hand) are
* `GameDecision.HIT`: Deciding to take another card, possibly risking exceeding 21.
* `GameDecision.STAND`: Deciding to stop being dealt cards, possibly risking the dealer exceeding your value.
* `GameDecision.SPLIT`: Split the initial hand into two separate independent hands and equalling the wager on the new hand as the first one.
* `GameDecision.DOUBLE`: Doubling the stake on the current hand, and hitting exactly one more time.

The possible states of a game are
* `GameState.LIVE`: The hand is still in-play and requires a decision on its next turn. 
* `GameState.STAND`: The hand has not lost, but its last play was a STAND, waiting for the outcome of the game.
* `GameState.WON`: The player beat the dealer. 
* `GameState.LOST`: The player lost to the dealer.
* `GameState.DRAW`: The player and dealer have the same value cards, or both have blackjack.

### Playing a Game
Suppose that a game is to be played with the following strategy:
* Always bet €10
* Split if the hand is splittable,
* Stand if the value is >= 17,
* Hit if the value is between 12 and 16,
* Double if the value is 10 or 11,
* Hit otherwise.

Also assume that the game is played with the following rules:
* Doubling After Splitting (DAS) is allowed,
* Splitting and hitting Aces is allowed,
* Re-splitting is allowed indefinitely,
* Dealer Hits on soft 17 (a.k.a H17),
* 6 Decks are used,
* Surrenter is not allowed,
* Only the original bet is lost on dealer blackjack.

```python
from blackjack.game import Blackjack
from blackjack.player import Player
from blackjack.utils import GameDecision

player = Player(stake=10)
game = Blackjack(
    players=[player],
    n_packs=6,
    double_after_split=True,
    hit_on_soft_17=True,
)

while not game.is_dealer_turn:
    hand = game.next_hand()
    
    if hand.is_splittable:
        decision = GameDecision.SPLIT
    elif hand.value() >= 17:
        decision = GameDecision.STAND
    elif 12 <= hand.value() < 17:
        decision = GameDecision.HIT
    elif 10 <= hand.value() < 12:
        decision = GameDecision.DOUBLE
    elif hand.value() < 10:
        decision = GameDecision.HIT
    else:
        raise ValueError('Hand value is not possible!')
    
    do_action = getattr(game, decision.value)
    do_action()

game.hit_dealer()

print(game)
print(f'Player Stake = €{player.stake}')
print(f'Player Profit = €{player.profit}')
```

<details>
<summary>Output</summary>

```
******************************
Dealer Hand: [J, T] (20)
(LOST) Player 0 Hand: [K, 9] (19)
(LOST) Player 0 Hand: [K, 8] (18)
Player Stake = €2
Player Profit = €-2
```
</details>

Alternatively, you can override the `strategy` and `get_stake` methods of `blackjack.Simulator.Simulation`, an example of which is given below.
If 
* a `basic_strategy` dict is provided, containing a mapping between all player-dealer combinations and the respective play,
* and the `strategy` method is _not_ overridden,

then the strategy will be played as per the provided strategy mapping. An example of this can be found in the `basic_strategy.py` script.

```python
from blackjack.Simulator import Simulation, GameDecision, Hand

class SomeStrategy(Simulation):
    def strategy(self, player_hand: Hand, dealer_hand: int) -> GameDecision:
        if player_hand.is_splittable:
            decision = GameDecision.SPLIT
        elif player_hand.value() >= 17:
            decision = GameDecision.STAND
        elif 12 <= player_hand.value() < 17:
            decision = GameDecision.HIT
        elif 10 <= player_hand.value() < 12:
            decision = GameDecision.DOUBLE
        elif player_hand.value() < 10:
            decision = GameDecision.HIT
        else:
            raise ValueError('Hand value is not possible!')
        return decision
    
    def get_stake(self) -> int:
        return self.initial_bet
    
sim = SomeStrategy(
        rounds_per_game=1,
        double_after_split=True,
        hit_on_soft_17=True,
        n_packs=6,
        initial_bank=1,
        initial_bet=1,
    )

sim.run()

print(sim.game)
print(f'Player Stake = €{sim.player.stake}')
print(f'Player Profit = €{sim.player.profit}')
```

<details>
<summary>Output</summary>

```
******************************
Dealer Hand: [5, 4] (9)
(LIVE) Player 0 Hand: [7, 6] (13)
Player Stake = €1
Player Profit = €0
```
</details>

### Generating Basic Strategy

Running a simulation assuming the above game rules and saving the output strategy can be achieved by running the following python script

```
gen_basic_strategy.py \
    --sample 400000 \
    --processes -1 \
    --generate \
    --double_after_split \
    --hit_on_soft_17 \
    --num_decks 6
```
This generates 400,000 game simulations on each possible player-dealer starting hand combination, and runs the process in parallel on the maximum allowed CPU cores on the decide being used. A specific number can also be desired instead of `-1` if one wishes. The rest of the arguments are self-explanatory and relate to the rules of the game.

This results in two `CSV` files named `basic_strategy.csv` and `basic_strategy_profit.csv`. The first file contains a tabulated list of decisions to be taken on each possible hand combination. 
The latter gives the expected value of the hand assuming €1 is played.

**NOTE**: Only DAS works at the time of writing.

## Optimal Strategy
unning the above command and generating the optimal strategies, we obtain the following mapping. This is identical to the one given by [The Wizard of Odds](https://wizardofodds.com/games/blackjack/strategy/4-decks/). 

| **Player \ Dealer** |  **2** |  **3** |  **4** |  **5** |  **6** |  **7** |  **8** |  **9** | **10** | **A**  |
|---------------------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|        **5**        | hit    | hit    | hit    | hit    | hit    | hit    | hit    | hit    | hit    |  hit   |
|        **6**        | hit    | hit    | hit    | hit    | hit    | hit    | hit    | hit    | hit    |  hit   |
|        **7**        | hit    | hit    | hit    | hit    | hit    | hit    | hit    | hit    | hit    |  hit   |
|        **8**        | hit    | hit    | hit    | hit    | hit    | hit    | hit    | hit    | hit    |  hit   |
|        **9**        | hit    | double | double | double | double | hit    | hit    | hit    | hit    |  hit   |
|        **10**       | double | double | double | double | double | double | double | double | hit    |  hit   |
|        **11**       | double | double | double | double | double | double | double | double | double | double |
|        **12**       | hit    | hit    | stand  | stand  | stand  | hit    | hit    | hit    | hit    |  hit   |
|        **13**       | stand  | stand  | stand  | stand  | stand  | hit    | hit    | hit    | hit    |  hit   |
|        **14**       | stand  | stand  | stand  | stand  | stand  | hit    | hit    | hit    | hit    |  hit   |
|        **15**       | stand  | stand  | stand  | stand  | stand  | hit    | hit    | hit    | hit    |  hit   |
|        **16**       | stand  | stand  | stand  | stand  | stand  | hit    | hit    | hit    | hit    |  hit   |
|        **17**       | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  |
|        **18**       | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  |
|        **19**       | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  |
|        **20**       | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  |
|        **21**       | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  |
|       **2,2**       | split  | split  | split  | split  | split  | split  | hit    | hit    | hit    |  hit   |
|       **3,3**       | split  | split  | split  | split  | split  | split  | hit    | hit    | hit    |  hit   |
|       **4,4**       | hit    | hit    | hit    | split  | split  | hit    | hit    | hit    | hit    |  hit   |
|       **5,5**       | double | double | double | double | double | double | double | double | hit    |  hit   |
|       **6,6**       | split  | split  | split  | split  | split  | hit    | hit    | hit    | hit    |  hit   |
|       **7,7**       | split  | split  | split  | split  | split  | split  | hit    | hit    | hit    |  hit   |
|       **8,8**       | split  | split  | split  | split  | split  | split  | split  | split  | split  | split  |
|       **9,9**       | split  | split  | split  | split  | split  | stand  | split  | split  | stand  | stand  |
|       **T,T**       | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  |
|       **A,2**       | hit    | hit    | hit    | double | double | hit    | hit    | hit    | hit    |  hit   |
|       **A,3**       | hit    | hit    | hit    | double | double | hit    | hit    | hit    | hit    |  hit   |
|       **A,4**       | hit    | hit    | double | double | double | hit    | hit    | hit    | hit    |  hit   |
|       **A,5**       | hit    | hit    | double | double | double | hit    | hit    | hit    | hit    |  hit   |
|       **A,6**       | hit    | double | double | double | double | hit    | hit    | hit    | hit    |  hit   |
|       **A,7**       | double | double | double | double | double | stand  | stand  | hit    | hit    |  hit   |
|       **A,8**       | stand  | stand  | stand  | stand  | double | stand  | stand  | stand  | stand  | stand  |
|       **A,9**       | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  |
|       **A,A**       | split  | split  | split  | split  | split  | split  | split  | split  | split  | split  |
|       **A,T**       | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  | stand  |
## Strategies
Players use various strategies to play BJ, some based on intuition, other based on basic strategy, 
and others try to beat the house by counting cards and raising the stake when there is a significant chance of a face card to show up,
hence a high change of the dealer to bust.

Ultimately, most strategies rely on playing the basic strategy or a variant of it, and only adjusting the stake when necessary.
What typically changes is the stake played on a hand based on the game state.
Thus our implementation of the strategy below simply inherit from the `simulator.Simulation` class
and override the `get_stake` method.
### Basic Strategy
Following the basic strategy is the simplest, yet most effective strategy were you to play blackjack without counting cards.
It is the assumption the casino makes when deciding on the edge, and the house edge is calculated based on this assumption.
The house edge of a game of BJ is typically less than 1%, at around 0.5%, making it one of the cheapest games in the casino for the player.
The `get_stake` method is very simple, where the same initial stake is always played.
```python
def get_stake(self) -> int:
    return self.initial_bet
```
Our simulated game of BJ uses rules that are mostly in favour of the player, and hence the house edge is estimated to be about 0.32%.

![monte_carlo_basic_strategy](https://github.com/DylanZammit/blackjack-simulator/blob/master/img/basic_strategy.jpg?raw=true)
### Martingale Strategy
A common strategy amongst gamblers is the martingale strategy, wherein a person bets €X, and if the wager is lost, the next wager is doubled to €2X. 
If this wager is also lost, the next wager will again be doubled to €4X, so on and so on. Once the player wins, they will recoup all their losses and gain an extra profit
over and above their previous losses. This strategy is a perfectly valid strategy, as a win is guaranteed with enough games played.
In practice however, this fails for two main reasons
* The assumption is that the player has enough money to finance their strategy, since wagers grow exponentially. This strategy essentially assumes that the player has infinite money.
* There are no maximum wagers on games. This is usually not the case in casinos, and most BJ tables enforce a maximum stake, rendering such a strategy useless.

Below is an example customer journey using the Martingale strategy. Losses are always recouped with a little extra profit,
resulting in an overall upward trend. However, this ends abruptly since eventually there will be enough losses which the customer
will not be able to finance since their bankroll is finite.

The `get_stake` method is implemented as follows.
```python
def get_stake(self) -> int:
    return min(2 ** self.n_consecutive_losses * self.initial_bet, self.bank)
```

![martingale_strategy](https://github.com/DylanZammit/blackjack-simulator/blob/master/img/martingale_strategy.jpg?raw=true)
### Hi-Lo Strategy
Card counting is less complicated than most people think. It does not involve perfect memorisation of all the previous hands that have come up, and certainly does not entail knowing exact probabilities of all outcomes of any combination of hands.
This misconception was popularised by movies such as Rain Man and 21.
However, most card-counting strategies assign multiple cards a value of 0, +1 or -1. 
The card counter must only keep a running total of these three values, and there is no need to remember information about the specific cards that showed up.
The simplest, and most popular card-counting strategy is the Hi-Lo strategy, which assigns
* a value of +1 to cards 2, 3, 4, 5, 6
* a value of  0 to cards 7, 8, 9
* a value of -1 to cards 10, J, Q, K, A.

Having a large positive count means that more small-value cards showed up than large value ones. 
In turn it means that there is a significant change that dealer hits on a large-value card and busts, leaving the player with an overall positive edge.
On the other hand, if the count is negative, it means that it is not advantageous for the player to wager a large amount since the dealer has a good chance of being dealt small-value hands, and hence comfortably fall in the range 17 to 21.

The number of decks in the show also influences the decision the player takes, since a larger number of decks introduces what is called variance.
A +2 count on a 2-deck game is much more significant than a 6-deck game, since the chances of a high-value count is still small due to the large pool of available cards in the shoe.
Instead of the raw count, card counters use the "True Count", which is simply the count divided by the number of decks being played.

In our simple implementation of this strategy, we will keep in mind the true count of the game at every iteration, and only when the true count is 2 or more,
we will place a large-value bet, linearly increasing with the true count. In other words
* for example for true counts of <= 1, we only stake €1, simply staying in the game and observing the count
* when the count is 2 we play a relatively large amount, like €100
* when the count is 3 we play €200
* when the count is >= 4 we play €300.

There are more complex variations of this strategy, where the basic strategy is dynamic depending on the true count.
Even with this simple implementation, the player is able to beat the house, with an edge of 0.5%.
Implementing more complex strategies allows the player to gain an even larger advantage.

The `get_stake` method is implemented as follows.
```python
def get_stake(self) -> int:
    betting_unit = self.bank // 100
    multiplier = 0 if self.game.is_time_to_shuffle else min(4, (self.game.true_count - 1))
    stake = max(self.initial_bet, multiplier * betting_unit)
    return stake
```
