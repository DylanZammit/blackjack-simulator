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
* 6 decks of cards are used
* Doubling after a split is allowed,
* Any number of splits are allowed
* dealer hits on soft 17.

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
Before generating the optimal strategy, i.e. the one that maximises the player's return and minimises the house edge, the rules of the game must be decided. During development, the following rules were assumed:
* Doubling After Splitting (DAS) is allowed,
* Splitting and hitting Aces is allowed,
* Re-splitting is allowed indefinitely,
* Dealer Hits on soft 17 (a.k.a H17),
* 6 Decks are used,
* Only the original bet is lost on dealer blackjack.

Running a simulation on such conditions and saving the output strategy can be achieved by running the following python script

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
|        **2**        | hit    | hit    | hit    | hit    | hit    | hit    | hit    | hit    | hit    |  hit   |
|        **3**        | hit    | hit    | hit    | hit    | hit    | hit    | hit    | hit    | hit    |  hit   |
|        **4**        | hit    | hit    | hit    | hit    | hit    | hit    | hit    | hit    | hit    |  hit   |
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
### Basic Strategy
### Martingale Strategy
### Hi-Lo Strategy
