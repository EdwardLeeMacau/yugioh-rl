import json
import os
import random

from abc import ABC, abstractmethod
from datetime import datetime
from enum import IntEnum
from itertools import combinations
from threading import Thread
from typing import List, Tuple

import accounts
from player import Player, GameState, Action

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
        if 'actions' in state and isinstance(state['actions'], list):
            return state['actions']

        match state['state']['phase']:
            case 0x1:
                return ['c', ]

            case 0x2:
                return ['c', ]

            case 0x4:
                # See: Duel.msg_handlers['select_tribute']
                if state['?'].get('requirement', None) == 'TRIBUTE':
                    min_cards = state['?']['min']

                    # ['1', '2', '3', '4', '5', '6', '7', ... ]
                    indices = list(map(str, range(1, len(state['?']['choices']) + 1)))

                    # [('1', '2'), ('1', '3'), ... ]
                    options = list(combinations(indices, min_cards))

                    # [('1 2'), ('1 3'), ... ]
                    options = list(map(lambda x: ' '.join(x), options))

                    return options

                # See: Duel.msg_handler['select_place']
                #
                # PLACE monster cards / spell cards
                # Auto-decidable. Not different in YGO04.
                if state['?'].get('requirement', None) == 'PLACE':
                    n = state['?']['min']

                    option = ' '.join(state['?']['choices'][:n])

                    return [option]

                options = []

                # usable = set(state['?']['summonable'] + state['?']['mset'] + state['?']['spsummon'])
                # options.extend(list(usable))

                # Perform face-up attack position summon. The place to summon is random selected.
                options.extend(list(map(lambda x: x + '\r\ns', state['?']['summonable'])))

                # Perform face-down defense position summon. The place to summon is random selected.
                options.extend(list(map(lambda x: x + '\r\nm', state['?']['mset'])))

                if state['?']['to_bp']:
                    options.append('b')

                if state['?']['to_ep']:
                    options.append('e')

                return options

            # See: Duel.msg_handlers['battle_attack']
            # See: Duel.msg_handlers['display_battle_menu']
            case 0x8:
                if state['?'].get('requirement', None) == 'SELECT':
                    min_cards = state['?']['min']

                    # ['1', '2', '3', '4', '5', '6', '7', ... ]
                    indices = list(map(str, range(1, len(state['?']['choices']) + 1)))

                    # [('1', '2'), ('1', '3'), ... ]
                    options = list(combinations(indices, min_cards))

                    # [('1 2'), ('1 3'), ... ]
                    options = list(map(lambda x: ' '.join(x), options))

                    return options

                # See: Duel.msg_handlers['select_option']
                # Options for battle phase
                if state['?'].get('requirement', None) == 'EFFECT':
                    return state['?']['choices']

                options = []

                # Perform attack. The target to attack is random selected.
                options.extend(list(map(lambda x: 'a\r\n' + x, state['?']['attackable'])))

                if state['?']['to_m2']:
                    options.append('m')

                if state['?']['to_ep']:
                    options.append('e')

                return options

            case 0x200:
                if state['?']['requirement'] != 'SELECT':
                    raise ValueError(f"requirement is not SELECT, but {state['?']['requirement']}")

                # See: Duel.msg_handlers['select_card']
                # !! Cannot hold more than 6 cards in hand
                min_cards = state['?']['min']

                # ['1', '2', '3', '4', '5', '6', '7', ... ]
                indices = list(map(str, range(1, len(state['?']['choices']) + 1)))

                # [('1', '2'), ('1', '3'), ... ]
                options = list(combinations(indices, min_cards))

                # [('1 2'), ('1 3'), ... ]
                options = list(map(lambda x: ' '.join(x), options))

                return options

            case _:
                return ['e', ]

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
