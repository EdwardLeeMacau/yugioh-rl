import random

from env.game import GameState, Action, Policy


class RandomPolicy(Policy):
    def react(self, state: GameState) -> Action:
        actions = self.list_valid_actions(state)
        return random.choice(actions)
