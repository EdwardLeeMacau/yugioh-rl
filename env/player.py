import io
import json
import os
import pysnooper

from itertools import combinations
from pprint import pformat
from telnetlib import Telnet
from typing import Dict, List, Tuple

from . import accounts
from .accounts import Account
from .state import StateMachine, GameState, Action


# Tee-Like object in Python:
# https://python-forum.io/thread-40226.html

class Player:
    """ Class to wrap the communication with the server. """
    _host: str
    _port: int
    _username: str
    _password: str

    _sm: StateMachine
    _state: GameState

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
        self._write(self._password.encode() + b'\r\n')

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

    # --------------------------------------------------------------------------
    # Actions for game initializing
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # Player's actions
    # --------------------------------------------------------------------------

    def list_valid_actions(self) -> Tuple[List[Action], List[Action]]:
        """ Decide an action from the valid actions. """
        return self._sm.list_valid_actions()

    def wait(self) -> Dict:
        """ Wait until the server ask player to decide an action.

        Assume that the server would send a message like this:

            [human-readable-texts]|<JSON-string>|

        Thus, the agent waits until the server send a message that starts and ends with "|".

        Returns
        -------
        embed : Dict
            Game state and valid actions in JSON format.
        """

        # Block I/O until the separator is found
        _ = self._read_until(b"|")

        # Load the JSON string until the separator is found
        embed = self._server.read_until(b"|")

        # Remove the separator and parse the JSON string
        embed = json.loads(embed[:-1])

        self._log.write(bytes(pformat(embed), 'utf-8'))
        self._log.flush()

        return embed

    def decode(self, embed: Dict) -> Tuple[bool, GameState]:
        """
        Returns
        -------
        terminated : bool
            Whether the game is over.

        state : GameState
            The state of the game.
        """
        # Check if the key 'terminated' is in the JSON string
        # 1: win, -1: lose, 0: draw
        if 'terminated' in embed:
            return True, {'score': embed['score']}

        # Auto deal with the PLACE requirement
        while 'actions' in embed and embed['actions']['requirement'] == 'PLACE':
            n = embed['actions']['min']
            response = ' '.join(embed['actions']['options'][:n])

            self._write(response.encode() + b'\r\n')
            embed = self.wait()

        if 'terminated' in embed:
            return True, {'score': embed['score']}

        self._sm = StateMachine.from_dict(embed['actions']) if 'actions' in embed else None
        self._state = embed['state']
        return False, embed['state']

    def step(self, action: Action) -> Tuple[bool, GameState]:
        if not self._sm.step(action):
            return False, self._state

        # Form a complete message to server
        self._write(self._sm.to_string().encode() + b'\r\n')

        # Wait for next decision
        return self.decode(self.wait())
