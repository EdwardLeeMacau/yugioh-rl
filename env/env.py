"""
env.py

Wrap the class Game as SinglePlayerEnv and MultiPlayerEnv.

SinglePlayerEnv: 1 learner vs 1 RandomAgent
 MultiPlayerEnv: 1 learner vs 1 learner
"""

import pysnooper

from datetime import datetime
from itertools import combinations
from multiprocessing import Process
from typing import Dict, List, Tuple

from .game import Action, Game, GameState, Player, Policy

Info = Dict

def game_loop(player: Player, policy: Policy) -> None:
    # For now we use threads to demonstrate the game loop for simplicity,
    # because wait_action() is a blocking I/O operation.

    terminated, state = player.decode(player.wait())
    while not terminated:
        actions = player.list_valid_actions()
        action = policy.react(state, actions)
        terminated, state = player.step(action)

    return

class SinglePlayerEnv:
    _game: Game
    _opponent: Policy
    _process: Process

    _state: Tuple[GameState, float, bool, bool, Info]

    def __init__(self, opponent: Policy):
        # Game content related resources.
        self._game = None
        self._opponent = opponent
        self._process = None

        self._state = None

    @property
    def player(self) -> Player:
        return self._game._player1

    def list_valid_actions(self) -> List[Action]:
        """ List all valid actions given the state. """
        return self.player.list_valid_actions()

    def reset(self, seed=None) -> Tuple[GameState, Info]:
        """ Reset the game.

        Arguments
        ---------
        seed: int | None
            The random seed for the game.
            ! NOT work because the randomness comes from both the game and the agent.

        Returns
        -------
        state: GameState
            The initial state of the game.

        info: Info
            Additional information.
        """
        # Halt the previous launched thread.
        if self._process is not None:
            self._process.terminate()

        # Assume that all resources are released after the instance
        # is no longer referenced by any variables.
        if self._game is not None:
            self._game.close()

        # Re-create the game instance.
        self._game = Game().start()

        # Launch a new thread for the opponent's decision making.
        self._process = Process(target=game_loop, args=(self._game._player2, self._opponent))
        self._process.start()

        # Wait until server acknowledges the player to make a decision.
        _, state = self.player.decode(self.player.wait())
        self._state = (state, 0.0, False, False, {})
        return state, {}

    def last(self) -> Tuple[GameState, float, bool, bool, Dict]:
        """ Return the last state, reward, termination, truncation, and info. """
        return self._state

    def step(self, action: Action) -> Tuple[GameState, float, bool, bool, Dict]:
        terminated, state = self.player.step(action)

        # Field 'score' exists only when the game is over.
        # +1.0 for win, -1.0 for lose, 0.0 for draw.
        reward = state.get('score', 0.0)

        self._state = (state, reward, terminated, False, {})
        return self._state

    def render(self):
        raise NotImplementedError
