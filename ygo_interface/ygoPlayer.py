import io
import json
import os
import random

from datetime import datetime
from enum import IntEnum
from itertools import combinations
from pprint import pprint, pformat
from telnetlib import Telnet
from typing import Dict, List, Tuple


Action = str
GameState = Dict

# Tee-Like object in Python:
# https://python-forum.io/thread-40226.html

class RockPaperScissorsAction(IntEnum):
    Rock = 1
    Paper = 2
    Scissors = 3

class YGOPlayer:
    # Public attributes
    username: str

    # Private attributes
    _log: io.StringIO
    _server: Telnet

    def __init__(self, host: str, port: int, username: str,  password: str) -> None:
        # create a log file
        os.makedirs('logs', exist_ok=True)
        # now = datetime.now().strftime('%Y%m%d-%H%M%S')
        self._log = open(f'logs/{username}.log', 'wb')

        # member fields
        self.username = username

        # create a connection to the server
        self._server = Telnet(host, port)

        # sign in to server
        self._read_until(b'\r\n')
        self._write(username.encode() + b'\r\n')
        self._read_until(b'\r\n')
        self._write(password.encode() + b'\r\n')

    def _write(self, msg: bytes) -> None:
        self._server.write(msg)

        # Debugging usage: print the message to the log file
        self._log.write(msg)
        self._log.flush()
        return None

    def _read_until(self, expected: bytes) -> bytes:
        msg = self._server.read_until(expected)

        # Debugging usage: print the message to the log file
        self._log.write(msg)
        self._log.flush()

        return msg

    def create_room(self) -> None:
        # create room
        self._read_until(b'\r\n')
        self._write(b'create\r\n')
        self._read_until(b'Enter ? to show all commands and room preferences\r\n')

        self._write(b'banlist unlimited\r\n')
        self._read_until(b"The banlist for this room was set to unlimited.\r\n")
        self._read_until(b'\r\n')

        self._write(b'finish\r\n')
        self._read_until(b"created a new duel room.\r\n")
        self._read_until(b'\r\n')

        # select team and deck
        self._write(b'move 1\r\n')
        self._read_until(b'You were moved into team 1.\r\n')
        self._read_until(b'\r\n')

        self._write(b'deck YGO04\r\n')
        self._read_until(b'Deck YGO04 loaded with 40 cards.\r\n')
        self._read_until(b'\r\n')

    def join(self, room_name: str) -> None:
        # join to the room
        self._read_until(b'\r\n')
        self._write(b'join ' + room_name.encode() + b'\r\n')
        self._read_until(b'Enter ? to show all commands and room preferences\r\n')

        # select team and deck
        self._write(b'move 2\r\n')
        self._read_until(b'You were moved into team 2.\r\n')
        self._read_until(b'\r\n')

        self._write(b'deck YGO04\r\n')
        self._read_until(b'Deck YGO04 loaded with 40 cards.\r\n')
        self._read_until(b'\r\n')

    def start_game(self):
        self._write(b'start\r\n')
        self._read_until(b'You start the duel.\r\n')

    def play_rock_paper_scissors(self, action: RockPaperScissorsAction):
        self._read_until(b"@abort to abort.\r\n")
        self._write( str(int(action)).encode() + b'\r\n')

    def who_first(self):
        self._read_until(b"@abort to abort.\r\n")
        self._write(b"1\r\n")

    def wait_action(self) -> Tuple[bool, GameState]:
        """ Wait until the server ask player to decide an action.

        Assume that the server would send a message like this:

            [human-readable-texts]|<JSON-string>|

        Thus, the agent waits until the server send a message that starts and ends with "|".

        Returns
        -------
        terminated : bool
            Whether the game is over.

        state : GameState
            The state of the game.
        """

        # Block I/O until the separator is found
        _ = self._read_until(b"|")

        # Load the JSON string until the separator is found
        embed = self._server.read_until(b"|")

        # Remove the separator and parse the JSON string
        embed = json.loads(embed[:-1])

        self._log.write(bytes(pformat(embed), 'utf-8'))
        self._log.flush()

        # Check if the key 'terminated' is in the JSON string
        if 'terminated' in embed:
            return True, {}

        return False, embed

    def finalize(self) -> None:
        self._server.close()
        self._log.close()

        return None

    def form_valid_actions(self, state: GameState) -> List[Action]:
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

    # !! TODO: Implement the policy here !!
    def react(self, state: GameState) -> Action:
        """ Decide an action based on the current state of the game. """
        actions = self.form_valid_actions(state)
        action = random.choice(actions)
        return action

    def interact(self, action: Action):
        self._write(str(action).encode() + b'\r\n')
