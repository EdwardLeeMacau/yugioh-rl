import io
import json
import os

from itertools import combinations
from pprint import pformat
from telnetlib import Telnet
from typing import Dict, List, Tuple

from . import accounts
from .accounts import Account


Action = str
GameState = Dict

# Tee-Like object in Python:
# https://python-forum.io/thread-40226.html

class Player:
    """ Class to wrap the communication with the server. """
    _host: str
    _port: int
    _username: str
    _password: str

    _state: GameState
    _action_queue: List[Action]

    _log: io.StringIO
    _server: Telnet

    def __init__(self) -> None:
        # Allocate an account.
        # A blocking call for simplicity.
        account = accounts.allocate()

        # create a log file
        os.makedirs('logs', exist_ok=True)
        self._log = open(f'logs/{account.username}.log', 'wb')

        # member fields
        self._host = account.host
        self._port = account.port
        self._username = account.username
        self._password = account.password

        self._state = {}
        self._action_queue = []

        # create a connection to the server
        self.open()

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

    def open(self) -> None:
        """ Open a connection to the server. """
        self._server = Telnet(self._host, self._port)

        self._read_until(b'\r\n')
        self._write(self.username.encode() + b'\r\n')
        self._read_until(b'\r\n')
        self._write(self.password.encode() + b'\r\n')

    def close(self) -> None:
        """ Free resources when the object is deleted. """
        if bool(self._server.sock) == True:
            self._server.close()

        if not self._log.closed:
            self._log.close()

        accounts.free(Account(
            host=self._host, port=self._port,
            username=self._username, password=self._password
        ))

    @property
    def username(self) -> str:
        return self._username

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

    def play_rock_paper_scissors(self, action):
        self._read_until(b"@abort to abort.\r\n")
        self._write(str(int(action)).encode() + b'\r\n')

    def first(self):
        self._read_until(b"@abort to abort.\r\n")
        self._write(b"1\r\n")

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
        # 1: win, -1: lose, 0: draw
        if 'terminated' in embed:
            return True, {'score': embed['score']}

        return False, embed

    def interact(self, action: Action):
        self._write(str(action).encode() + b'\r\n')
