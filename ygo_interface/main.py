from telnetlib import Telnet
from ygoPlayer import YGOPlayer
from ygoGame import YGOGame

def main():
    HOST = "cubone.csie.org"
    PORT = 4000

    player1 = YGOPlayer(HOST, PORT, "player7", "player7")
    player2 = YGOPlayer(HOST, PORT, "player8", "player8")

    game = YGOGame(player1, player2)
    game.start()

if __name__ == '__main__':
    main()