import io
import json
import os
from pprint import pformat, pprint
from telnetlib import Telnet
from typing import Dict, List, Tuple

from . import accounts
from .accounts import Account
from .state import Action, GameState, StateMachine

# Tee-Like object in Python:
# https://python-forum.io/thread-40226.html

class Player:
    """ Class to wrap the communication with the server. """
    _account: Account

    _sm: StateMachine
    _state: GameState

    _log: io.StringIO
    _server: Telnet

    def __init__(self) -> None:
        # Allocate an account.
        # A blocking call for simplicity.
        self._account = accounts.allocate()

        # create a connection to the server
        self.open()

    def _write(self, msg: bytes) -> None:
        self._server.write(msg)
        return None

    def _read_until(self, expected: bytes) -> bytes:
        msg = self._server.read_until(expected)
        return msg

    def open(self) -> None:
        """ Open a connection to the server. """
        self._server = Telnet(self._account.host, self._account.port)

        self._read_until(b'\r\n')
        self._write(self._account.username.encode() + b'\r\n')
        self._read_until(b'\r\n')
        self._write(self._account.password.encode() + b'\r\n')

    def close(self) -> None:
        """ Free resources when the object is deleted. """
        if bool(self._server.sock) == True:
            self._server.close()

        accounts.free(Account(
            host=self._account.host, port=self._account.port,
            username=self._account.username, password=self._account.password
        ))

    @property
    def username(self) -> str:
        return self._account.username

    # --------------------------------------------------------------------------
    # Actions for game initializing
    # --------------------------------------------------------------------------

    def create_room(self, advantages: Dict) -> None:
        # create room
        self._read_until(b'\r\n')
        self._write(b'create\r\n')
        self._read_until(b'Enter ? to show all commands and room preferences\r\n')

        self._write(b'banlist unlimited\r\n')
        self._read_until(b"The banlist for this room was set to unlimited.\r\n")
        self._read_until(b'\r\n')

        self._write(f'lifepoints 1 {advantages["player1"].get("lifepoints", 8000)}\r\n'.encode())
        self._read_until(b'\r\n')

        self._write(f'lifepoints 2 {advantages["player2"].get("lifepoints", 8000)}\r\n'.encode())
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

    def list_valid_actions(self) -> Tuple[List[Action],
                                          List[Dict[Action, str]]]:
        """ Decide an action from the valid actions. """
        return self._sm.list_valid_actions() if self._sm is not None else ([], {})

    def last_option(self) -> Action | None:
        return self._sm.last_option() if self._sm is not None else None

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
        return embed

    def decode_server_msg(self, embed: Dict) -> Tuple[bool, Dict, Dict | None]:
        """
        Returns
        -------
        terminated : bool
            Whether the game is over.

        state : Dict
            The state in Dict format.

        actions : Dict | None
            The valid actions in Dict format.
        """
        return 'terminated' in embed, embed.get('state', None), embed.get('actions', None)

    def step(
            self,
            action: Action
        ) -> Tuple[bool, GameState, str | None]:
        """
        Returns
        -------
        terminated : bool
            Whether the game is over.

        state: GameState
            The state of the game.

        action: str | None
            The concrete action sent to the server.
            None if the step is only the part of the valid action.
        """
        if not self._sm.step(action):
            return False, self._state, None

        # Form a complete message to server
        action = self._sm.to_string()
        self._write(action.encode() + b'\r\n')

        while True:
            # Wait for next decision
            terminated, state, action = self.decode_server_msg(self.wait())

            # Auto deal with the PLACE requirement
            if action is None or action['requirement'] != 'PLACE':
                break

            n = action['min']
            response = ' '.join(action['options'][:n])
            self._write(response.encode() + b'\r\n')

        self._sm = StateMachine.from_dict(action)
        self._state = state

        return terminated, state, action
