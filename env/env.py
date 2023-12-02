"""
env.py

Wrap the class Game as SinglePlayerEnv and MultiPlayerEnv.

SinglePlayerEnv: 1 learner vs 1 RandomAgent
 MultiPlayerEnv: 1 learner vs 1 learner
"""

import json
import os
from datetime import datetime
from threading import Event, Thread
from typing import Dict, List, Tuple

from .game import Action, Game, GameState, Player, Policy

Info = Dict

def game_loop(event: Event, player: Player, agent: Policy) -> None:
    # For now we use threads to demonstrate the game loop for simplicity,
    # because wait_action() is a blocking I/O operation.

    while not event.is_set():
        terminated, state = player.wait_action()
        if terminated:
            break

        action = agent.react(state)
        player.interact(action)

    return

class SinglePlayerEnv:
    _game: Game
    _opponent: Policy
    _thread: Thread

    def __init__(self, opponent: Policy):
        self._game = None
        self._opponent = opponent
        self._thread = None
        self._event = Event()

    def reset(self, seed=None):
        """ Reset the game.

        Arguments
        ---------
        seed: int | None
            The random seed for the game.
            ! NOT work because the randomness comes from both the game and the agent.
        """
        # Try to halt the previous launched thread.
        self._event.set()
        if self._thread is not None:
            self._thread.join()

        # Assume that all resources are released after the instance
        # is no longer referenced by any variables.
        if self._game is not None:
            self._game.close()

        self._game = Game().start()
        self._event.clear()
        self._thread = Thread(target=game_loop, args=(self._event, self._game._player2, self._opponent))
        self._thread.start()

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