import json
import os
import random

from abc import ABC, abstractmethod
from datetime import datetime
from enum import IntEnum
from itertools import combinations
from threading import Thread
from typing import List, Tuple

from . import accounts
from .player import Player, GameState, Action

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
    def list_valid_actions(self, state: GameState) -> List[Action]:
        """ Decide an action from the valid actions. """
        options = []

        match state['?'].get('requirement', None):
            # See: Duel.msg_handler['select_*']
            case 'SELECT' | 'TRIBUTE':
                if (n := state['?'].get('foreach', None)) is None:
                    n = state['?']['min']

                match state['?'].get('type', 'indices'):
                    case 'spec':
                        # 'type': spec
                        #
                        # Return card specs to select, assume `n` is 1
                        # >>> ['h3', 'h4', 's5', ... ]
                        options = state['?']['choices']
                    case 'indices':
                        # 'type': indices
                        #
                        # Return indices of cards to select
                        # >>> [('1 2'), ('1 3'), ... ]
                        indices = list(map(str, range(1, len(state['?']['choices']) + 1)))
                        options = list(combinations(indices, n))
                        options = list(map(lambda x: ' '.join(x), options))
                    case _:
                        raise ValueError(f"unknown type {state['?']['type']}")

            # See: Duel.msg_handler['select_place']
            #
            # PLACE monster cards / spell cards
            # Auto-decidable. Not different in YGO04.
            case 'PLACE':
                n = state['?']['min']
                options = [' '.join(state['?']['choices'][:n])]

            # See: Duel.msg_handler['idle_action']
            case 'IDLE':
                # Perform face-up attack position summon. The place to summon is random selected.
                options.extend(list(map(lambda x: x + '\r\ns', state['?']['summonable'])))

                # Perform face-down defense position summon. The place to summon is random selected.
                options.extend(list(map(lambda x: x + '\r\nm', state['?']['mset'])))

                # Perform set spell/trap card. The place to set is random selected.
                # options.extend(list(map(lambda x: x + '\r\nt', state['?']['set'])))

                # Perform re-position.
                options.extend(list(map(lambda x: x + '\r\nr', state['?']['repos'])))

                # Perform special summon. The place to summon is random selected.
                options.extend(list(map(lambda x: x + '\r\nc', state['?']['spsummon'])))

                if state['?']['to_bp']:
                    options.append('b')

                if state['?']['to_ep']:
                    options.append('e')

            # See: Duel.msg_handlers['select_option']
            # Options for battle phase
            case 'EFFECT':
                return state['?']['choices']

            case 'BATTLE':
                # Perform attack. The target to attack is random selected.
                options.extend(list(map(lambda x: 'a\r\n' + x, state['?']['attackable'])))

                # Perform activate. The card to activate is random selected.
                options.extend(list(map(lambda x: 'c\r\n' + x, state['?']['activatable'])))

                if state['?']['to_m2']:
                    options.append('m')

                if state['?']['to_ep']:
                    options.append('e')

            case _:
                raise ValueError(f"unknown requirement {state['?'].get('requirement', None)}")

        return options

    @abstractmethod
    def react(self, state: GameState) -> Action:
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
