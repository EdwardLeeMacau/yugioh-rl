import pysnooper
from collections import defaultdict
from typing import Dict, List, Tuple

Action = str
GameState = Dict


class StateMachine:
    _current: GameState
    _queue: List[Action]

    def __init__(self, state_dict: GameState) -> None:
        self._current = state_dict
        self._queue = []

    def list_valid_actions(self, spec=False) -> Tuple[List[Action],
                                                      List[Dict[Action, str]]]:
        """ List valid actions.

        Parameters
        ----------
        spec : bool
            If True, return card specs instead of codes. Default: False.
        """
        match self._current.get('requirement', None):
            case 'BATTLE' | 'IDLE':
                return (self._current['options'], {})

            case 'BATTLE_ACTION' | 'IDLE_ACTION':
                action = self._queue[-1]
                target = {
                    card[1] : card[0] for card in self._current['targets'][action]
                }

                return ([], target, )

            case 'EFFECT':
                breakpoint()
                actions.append(self._current['effect'])

            case 'SELECT' | 'TRIBUTE' | 'YESNO' | "ANNOUNCE_RACE":
                breakpoint()
                match self._current.get('type', 'indices'):
                    case 'spec':
                        # 'type': spec
                        #
                        # Return card specs to select, assume `n` is 1
                        # >>> ['h3', 'h4', 's5', ... ]
                        cards = self._current['choices']
                    case 'indices':
                        # 'type': indices
                        #
                        # Return indices of cards to select
                        # >>> ['1', '2', ... ]
                        actions  = set(map(str, range(1, len(self._current['choices']) + 1)))
                        actions -= set(self._queue)
                        actions  = list(actions)

            case _:
                raise ValueError(f"Unknown requirement: {self._current.get('requirement', None)}")

        return (actions, cards)

    def to_string(self) -> str:
        if self._current.get('requirement', None) in ('SELECT', 'TRIBUTE'):
            return ' '.join(self._queue)

        return '\r\n'.join(self._queue)

    def step(self, action: Action) -> bool:
        match self._current.get('requirement', None):
            case 'BATTLE':
                self._queue.append(action)

                if action in ('m', 'e'):
                    return True

                self._current['requirement'] = 'BATTLE_ACTION'
                return False

            case 'BATTLE_ACTION' | 'IDLE_ACTION':
                self._queue.insert(0, action)
                return True

            case 'EFFECT':
                self._queue.append(action)
                return True

            case 'IDLE':
                self._queue.append(action)

                if action in ('b', 'e'):
                    return True

                self._current['requirement'] = 'IDLE_ACTION'
                return False

            case 'SELECT' | 'TRIBUTE' | 'YESNO' | "ANNOUNCE_RACE":
                self._queue.append(action)
                return len(self._queue) >= self._current.get('foreach', self._current['min'])

            case _:
                raise ValueError(f"Unknown requirement: {self._current.get('requirement', None)}")

    @classmethod
    def from_dict(cls, state_dict: Dict) -> 'StateMachine':
        return cls(state_dict)
