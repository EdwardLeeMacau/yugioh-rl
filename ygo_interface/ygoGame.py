from ygoPlayer import YGOPlayer
from utils import play_rock_paper_scissors

class YGOGame:
    def __init__(self, player1: YGOPlayer, player2: YGOPlayer) -> None:
        self.player1 = player1
        self.player2 = player2
    
    def start(self) -> None:
        self.player1.create_room()
        self.player2.join(self.player1.username)
        self.player1.start_game()

        who_first = play_rock_paper_scissors(self.player1, self.player2)

        first_player = self.player1 if who_first else self.player2
        second_player = self.player2 if who_first else self.player1

        first_player.server.interact()