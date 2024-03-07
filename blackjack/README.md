# Blackjack Package
The `blackjack` package contains various scripts containing classes required for a game of blackjack, following the below logic
* A `game.Blackjack` can contain at least one
* `player.Player`, each of which having at least one
* `hand.Hand`, each of which containing at least two
* `card.Card`, which are of the following possible ranks: `A23456789TJQK`.

The `utils.py` contains common functions and variables, most importantly two `Enum` classes: `utils.GameDecision` and `utils.GameState`.