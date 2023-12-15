import random
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, List, Tuple

from .player import Action, GameState, Player


class Policy(ABC):
    @abstractmethod
    def react(
            self,
            state: GameState,
            options: List[Action],
            targets: Dict[Action, List[Action]]
        ) -> Action:
        raise NotImplementedError

RACES = [
    "Warrior",
    "Spellcaster",
    "Fairy",
    "Fiend",
    "Zombie",
    "Machine",
    "Aqua",
    "Pyro",
    "Rock",
    "Winged Beast",
    "Plant",
    "Insect",
    "Thunder",
    "Dragon",
    "Beast",
    "Beast-Warrior",
    "Dinosaur",
    "Fish",
    "Sea Serpent",
    "Reptile",
    "Psychic",
    "Divine-Beast",
    "Creator God",
    "Wyrm",
    "Cyberse",
]

# List of all cards in the YGO04 format.
# Use for assigning the cards into the one-hot encoding / multi-hot encoding.
DECK = [
    None,       # empty space
    72989439,
    77585513,
    18036057,
    63749102,
    88240808,
    33184167,
    39507162,
    71413901,
    76922029,
    74131780,
    78706415,
    79575620,
    23205979,
    8131171,
    19613556,
    32807846,
    55144522,
    42829885,
    17375316,
    4031928,
    45986603,
    69162969,
    71044499,
    72302403,
    5318639,
    70828912,
    29401950,
    53582587,
    56120475,
    60082869,
    83555666,
    97077563,
    7572887,
    74191942,
    44095762,
    31560081,
    73915051,
    73915052,   # Scapegoat (should map to the same encoding)
    73915053,   # Scapegoat
    73915054,   # Scapegoat
    73915055,   # Scapegoat
    0,          # hidden card
]

# Map card ID to the index of the one-hot encoding / multi-hot encoding.
CARDS2IDX = {card: i for i, card in enumerate(DECK)}

# Scapegoat token (should map to the same encoding)
CARDS2IDX[73915052] = CARDS2IDX[73915051]
CARDS2IDX[73915053] = CARDS2IDX[73915051]
CARDS2IDX[73915054] = CARDS2IDX[73915051]
CARDS2IDX[73915055] = CARDS2IDX[73915051]

# Enumerate actions
TARGETS = [
    *DECK,
    *RACES,
]

OPTIONS = [
    "e", # enter end phase
    "s", # summon this card in face-up attack position
    "m", # summon this card in face-down defense position/ enter main phase
    "t", # set this card (Trap/Magic)
    "v", # activate this card
    "c", # special summon this card / activate effect during battle phase
    "b", # enter battle phase
    "y", # applies the effect
    "n", # does not apply the effect
    "a", # attack
    "r", # reposition
    '1', # select option of Don Zaloog
    '2',
    '3', # FACEUP.DEFENSE
    '4',
]

# DONT modify the order of the actions.
POSSIBLE_ACTIONS = [
    *TARGETS,
    *OPTIONS,
]

class RockPaperScissorsAction(IntEnum):
    Rock = 1
    Paper = 2
    Scissors = 3

class Game:
    _players: List[Player]
    _advantages: Dict

    def __init__(self, advantages: Dict) -> None:
        self._players = [Player(), Player()]
        self._advantages = advantages

    @staticmethod
    def determine_first_player(p1: Player, p2: Player) -> Tuple[Player]:
        """ Determine the first player by flipping a coin. """
        a1, a2 = RockPaperScissorsAction.Rock, RockPaperScissorsAction.Scissors
        a1, a2 = (a1, a2) if (res := random.random() < 0.5) else (a2, a1)

        p1.play_rock_paper_scissors(a1)
        p2.play_rock_paper_scissors(a2)

        return (p1, p2) if res else (p2, p1)

    def close(self) -> None:
        for p in self._players:
            p.close()

        return None

    def start(self) -> 'Game':
        self._players[0].create_room(self._advantages)
        self._players[1].join(self._players[0].username)
        self._players[0].start_game()

        p1, _ = self.determine_first_player(*self._players)
        p1.first()

        return self
