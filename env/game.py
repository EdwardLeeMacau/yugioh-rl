import random

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List, Tuple

from env import accounts
from env.player import Player, GameState, Action

class RockPaperScissorsAction(IntEnum):
    Rock = 1
    Paper = 2
    Scissors = 3

def determine_first_player(p1: Player, p2: Player) -> Tuple[Player]:
    """ Determine the first player by flipping a coin. """
    a1, a2 = RockPaperScissorsAction.Rock, RockPaperScissorsAction.Scissors
    a1, a2 = (a1, a2) if (res := random.random() < 0.5) else (a2, a1)

    p1.play_rock_paper_scissors(a1)
    p2.play_rock_paper_scissors(a2)

    return (p1, p2) if res else (p2, p1)

class Policy(ABC):
    @abstractmethod
    def react(self, state: GameState, actions: List[Action]) -> Action:
        raise NotImplementedError

class Game:
    _player1: Player | None
    _player2: Player | None

    # FIXME: Allocate accounts from the same game server.
    def __init__(self) -> None:
        self._player1 = Player()
        self._player2 = Player()

    def close(self) -> None:
        if self._player1 is not None:
            self._player1.close()

        if self._player2 is not None:
            self._player2.close()

        return None

    def start(self) -> 'Game':
        self._player1.create_room()
        self._player2.join(self._player1._username)
        self._player1.start_game()

        p1, _ = determine_first_player(self._player1, self._player2)
        p1.first()

        return self
