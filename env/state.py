from typing import Dict, List
from itertools import combinations

Action = str
GameState = Dict


class StateMachine:
    _current: GameState
    _queue: List[Action]

    def __init__(self, state_dict: GameState) -> None:
        self._current = state_dict
        self._queue = []

    def list_valid_actions(self) -> List[Action]:
        options: List[Action] = []

        match self._current.get('requirement', None):
            case 'BATTLE':
                options.extend(self._current['attackable'])
                options.extend(self._current['activatable'])
                options = list(set(options))

                if self._current['to_m2']:
                    options.append('m')

                if self._current['to_ep']:
                    options.append('e')

            case 'BATTLE_ACTION':
                if self._queue[-1] in self._current['attackable']:
                    options.append('a')

                if self._queue[-1] in self._current['activatable']:
                    options.append('c')

            case 'EFFECT':
                options.append(self._current['effect'])

            case 'IDLE':
                options.extend(self._current['summonable'])
                options.extend(self._current['mset'])
                options.extend(self._current['repos'])
                options.extend(self._current['spsummon'])
                options.extend(self._current['set'])
                options.extend(self._current['activate'])

                options = list(set(options))

                if self._current['to_bp']:
                    options.append('b')

                if self._current['to_ep']:
                    options.append('e')

            case 'IDLE_ACTION':
                if self._queue[-1] in self._current['summonable']:
                    options.append('s')

                if self._queue[-1] in self._current['mset']:
                    options.append('m')

                if self._queue[-1] in self._current['repos']:
                    options.append('r')

                if self._queue[-1] in self._current['spsummon']:
                    options.append('c')

                if self._queue[-1] in self._current['set']:
                    options.append('t')

                if self._queue[-1] in self._current['activate']:
                    options.append('v')

            case 'SELECT' | 'TRIBUTE' | 'YESNO' | "ANNOUNCE_RACE":
                match self._current.get('type', 'indices'):
                    case 'spec':
                        # 'type': spec
                        #
                        # Return card specs to select, assume `n` is 1
                        # >>> ['h3', 'h4', 's5', ... ]
                        options = self._current['choices']
                    case 'indices':
                        # 'type': indices
                        #
                        # Return indices of cards to select
                        # >>> ['1', '2', ... ]
                        options  = set(map(str, range(1, len(self._current['choices']) + 1)))
                        options -= set(self._queue)
                        options  = list(options)

        return options

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
