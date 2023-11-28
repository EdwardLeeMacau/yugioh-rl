from telnetlib import Telnet
from enum import IntEnum

class RockPaperScissorsAction(IntEnum):
    Rock = 1
    Paper = 2 
    Scissors = 3

class YGOPlayer:
    def __init__(self, host: str, port: int, username: str,  password: str) -> None:
        self.username = username
        # create a connection to the server
        self.server = Telnet(host, port)

        # sign in to server
        self.server.read_until(b'\r\n')
        self.server.write(username.encode() + b'\r\n')
        self.server.read_until(b'\r\n')
        self.server.write(password.encode() + b'\r\n')
    
    def create_room(self) -> None:
        # create room
        self.server.read_until(b'\r\n')
        self.server.write(b'create\r\n')
        self.server.read_until(b'Enter ? to show all commands and room preferences\r\n')

        self.server.write(b'banlist unlimited\r\n')
        self.server.read_until(b"The banlist for this room was set to unlimited.\r\n")
        self.server.read_until(b'\r\n')

        self.server.write(b'finish\r\n')
        self.server.read_until(b"created a new duel room.\r\n")
        self.server.read_until(b'\r\n')

        # select team and deck
        self.server.write(b'move 1\r\n')
        self.server.read_until(b'You were moved into team 1.\r\n')
        self.server.read_until(b'\r\n')

        self.server.write(b'deck YGO04\r\n')
        self.server.read_until(b'Deck YGO04 loaded with 40 cards.\r\n')
        self.server.read_until(b'\r\n')
    
    def join(self, room_name: str) -> None:
        # join to the room
        self.server.read_until(b'\r\n')
        self.server.write(b'join ' + room_name.encode() + b'\r\n')
        self.server.read_until(b'Enter ? to show all commands and room preferences\r\n')

        # select team and deck
        self.server.write(b'move 2\r\n')
        self.server.read_until(b'You were moved into team 2.\r\n')
        self.server.read_until(b'\r\n')

        self.server.write(b'deck YGO04\r\n')
        self.server.read_until(b'Deck YGO04 loaded with 40 cards.\r\n')
        self.server.read_until(b'\r\n')

    def start_game(self):
        self.server.write(b'start\r\n')
        self.server.read_until(b'You start the duel.\r\n')
    
    def play_rock_paper_scissors(self, action: RockPaperScissorsAction):
        self.server.read_until(b"@abort to abort.\r\n")
        self.server.write( str(int(action)).encode() + b'\r\n')
    
    def who_first(self):
        self.server.read_until(b"@abort to abort.\r\n")
        self.server.write(b"1\r\n") 