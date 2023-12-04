"""
env.py

Wrap the class Game as SinglePlayerEnv and MultiPlayerEnv.

SinglePlayerEnv: 1 learner vs 1 RandomAgent
 MultiPlayerEnv: 1 learner vs 1 learner
"""

from datetime import datetime
from itertools import combinations
from multiprocessing import Process
from typing import Dict, List, Tuple

from .game import Action, Game, GameState, Player, Policy

Info = Dict

def game_loop(player: Player, agent: Policy) -> None:
    # For now we use threads to demonstrate the game loop for simplicity,
    # because wait_action() is a blocking I/O operation.

    while True:
        terminated, state = player.wait_action()
        if terminated:
            break

        actions = player.list_valid_actions(state)
        action = agent.react(state, actions)
        player.interact(action)

    return

class SinglePlayerEnv:
    _game: Game
    _opponent: Policy
    _process: Process

    def __init__(self, opponent: Policy):
        self._game = None
        self._opponent = opponent
        self._process = None

    def list_valid_actions(self, state: GameState) -> List[Action]:
        """ List all valid actions given the state. """
        return self._game._player1.list_valid_actions(state)

    def reset(self, seed=None):
        """ Reset the game.

        Arguments
        ---------
        seed: int | None
            The random seed for the game.
            ! NOT work because the randomness comes from both the game and the agent.
        """
        # Try to halt the previous launched thread.
        # self._event.set()
        if self._process is not None:
            self._process.terminate()

        # Assume that all resources are released after the instance
        # is no longer referenced by any variables.
        if self._game is not None:
            self._game.close()

        self._game = Game().start()
        self._process = Process(target=game_loop, args=(self._game._player2, self._opponent))
        self._process.start()

    def last(self) -> Tuple[GameState, float, bool, bool, Dict]:
        """ Return the last state, reward, termination, truncation, and info. """
        terminated, state = self._game._player1.wait_action()

        # Field 'score' exists only when the game is over.
        # +1.0 for win, -1.0 for lose, 0.0 for draw.
        reward = state.get('score', 0.0)

        return state, reward, terminated, False, {}

    def step(self, action: Action):
        self._game._player1.interact(action)

    def render(self):
        raise NotImplementedError
