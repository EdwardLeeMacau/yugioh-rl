import json
import os
from datetime import datetime
from threading import Thread
from typing import List, Tuple

from utils import play_rock_paper_scissors
from ygoPlayer import YGOPlayer, GameState, Action


class YGOGame:
    # Public attributes
    ...

    # Private attributes
    _player1: YGOPlayer
    _player2: YGOPlayer
    _thread1: Thread
    _thread2: Thread

    def __init__(self, player1: YGOPlayer, player2: YGOPlayer) -> None:
        start = datetime.now().strftime('%Y%m%d-%H%M%S')
        def game_loop(player: YGOPlayer) -> None:
            trajectories: List[Tuple[GameState, Action]] = []
            while True:
                terminated, state = player.wait_action()
                if terminated:
                    break

                action = player.react(state)
                trajectories.append((state, action))
                player.interact(action)

            # Save the trajectories
            trajectories.append((state, None))
            with open(os.path.join('logs', f'{start}-{player.username}.json'), 'w') as f:
                json.dump(trajectories, f, indent=4)

            return

        self._player1 = player1
        self._player2 = player2
        self._thread1 = Thread(target=game_loop, args=(self._player1,))
        self._thread2 = Thread(target=game_loop, args=(self._player2,))

    def start(self) -> None:
        self._player1.create_room()
        self._player2.join(self._player1.username)
        self._player1.start_game()

        who_first = play_rock_paper_scissors(self._player1, self._player2)

        first_player = self._player1 if who_first else self._player2
        second_player = self._player2 if who_first else self._player1

        first_player.who_first()

        # -------------------------- Game Loop --------------------------

        self._thread1.start()
        self._thread2.start()

        self._thread1.join()
        self._thread2.join()

        # ------------------------ Game Loop End ------------------------

        self._player1.finalize()
        self._player2.finalize()

