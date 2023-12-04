import random
from typing import List

from env.game import GameState, Action, Policy


class RandomPolicy(Policy):
    def react(self, state: GameState, actions: List[Action]) -> Action:
        return random.choice(actions)
