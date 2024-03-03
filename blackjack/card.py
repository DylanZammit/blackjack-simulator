from typing import Union


class Card:

    deck = 'A23456789TJQK'

    def __init__(self, rank: str):
        if rank == '10':
            rank = 'T'
        elif rank in ['1', '11']:
            rank = 'A'
        else:
            rank = str(rank)

        assert rank in Card.deck, f'{rank} not found'
        self.rank = rank

    def __repr__(self):
        return self.rank

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.value == other.value
        elif isinstance(other, str):
            if other in Card.deck:
                return self.value == Card(other).value
            else:
                raise TypeError(f'Unknown card rank {other}')
        return False

    def __gt__(self, other):
        if isinstance(other, Card):
            return self.value > other.value
        elif isinstance(other, str):
            if other in Card.deck:
                return self.value > Card(other).value
            else:
                raise TypeError(f'Unknown card rank {other}')
        else:
            raise TypeError(f'Cannot add {type(other)}')

    @property
    def value(self) -> Union[int, tuple[int, int]]:
        if self.rank == 'A':
            return 1, 11
        if self.rank in 'TJQK':
            return 10

        return int(self.rank)

    @property
    def high_low_count(self):
        if isinstance(self.value, tuple):
            return -1

        if self.value <= 6:
            return 1
        elif 7 <= self.value <= 9:
            return 0
        else:
            return -1
