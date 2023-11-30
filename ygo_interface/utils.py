import random
from ygoPlayer import YGOPlayer, RockPaperScissorsAction

def determine_winner(player1_action: RockPaperScissorsAction, player2_action: RockPaperScissorsAction) -> bool | None:
    # True: player1 win
    # False: player2 win
    # None: draw

    if player1_action == player2_action:
        return None
    elif player1_action == RockPaperScissorsAction.Rock:
        if player2_action == RockPaperScissorsAction.Scissors:
            return True
        else:
            return False
    elif player1_action == RockPaperScissorsAction.Paper:
        if player2_action == RockPaperScissorsAction.Rock:
            return True
        else:
            return False
    elif player1_action == RockPaperScissorsAction.Scissors:
        if player2_action == RockPaperScissorsAction.Paper:
            return True
        else:
            return False

def play_rock_paper_scissors(player1: YGOPlayer, player2: YGOPlayer) -> bool:
    # Play rock paper scissors utils players's outcome difference
    while True:
        player1_action = random.choice(list(RockPaperScissorsAction))
        player2_action = random.choice(list(RockPaperScissorsAction))

        res = determine_winner(player1_action, player2_action)

        if res != None:
            player1.play_rock_paper_scissors(player1_action)
            player2.play_rock_paper_scissors(player2_action)
            break

    return res


