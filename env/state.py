import pysnooper
from typing import Dict, List, Tuple

Action = str
GameState = Dict


class StateMachine:
    _current: GameState
    _queue: List[Action]

    def __init__(self, state_dict: GameState) -> None:
        self._current = state_dict
        self._queue = []

    def list_valid_actions(self) -> Tuple[List[Action], List[Action]]:
        actions, cards = [], []

        match self._current.get('requirement', None):
            case 'BATTLE':
                cards.extend(self._current['attackable'])
                cards.extend(self._current['activatable'])
                cards = list(set(cards))

                if self._current['to_m2']:
                    actions.append('m')

                if self._current['to_ep']:
                    actions.append('e')

            case 'BATTLE_ACTION':
                if self._queue[-1] in self._current['attackable']:
                    actions.append('a')

                if self._queue[-1] in self._current['activatable']:
                    actions.append('c')

            case 'EFFECT':
                actions.append(self._current['effect'])

            case 'IDLE':
                cards.extend(self._current['summonable'])
                cards.extend(self._current['mset'])
                cards.extend(self._current['repos'])
                cards.extend(self._current['spsummon'])
                cards.extend(self._current['set'])
                cards.extend(self._current['activate'])

                cards = list(set(cards))

                if self._current['to_bp']:
                    actions.append('b')

                if self._current['to_ep']:
                    actions.append('e')

            case 'IDLE_ACTION':
                if self._queue[-1] in self._current['summonable']:
                    actions.append('s')

                if self._queue[-1] in self._current['mset']:
                    actions.append('m')

                if self._queue[-1] in self._current['repos']:
                    actions.append('r')

                if self._queue[-1] in self._current['spsummon']:
                    actions.append('c')

                if self._queue[-1] in self._current['set']:
                    actions.append('t')

                if self._queue[-1] in self._current['activate']:
                    actions.append('v')

            case 'SELECT' | 'TRIBUTE' | 'YESNO' | "ANNOUNCE_RACE":
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

            case 'BATTLE_ACTION':
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

            case 'IDLE_ACTION':
                self._queue.append(action)
                return True

            case 'SELECT' | 'TRIBUTE' | 'YESNO' | "ANNOUNCE_RACE":
                self._queue.append(action)
                return len(self._queue) >= self._current.get('foreach', self._current['min'])

    @classmethod
    def from_dict(cls, state_dict: Dict) -> 'StateMachine':
        return cls(state_dict)
