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

        Returns
        -------
        options : List[Action]
            List of valid actions.

        targets : List[Dict[Action, str]]
            List of valid targets in pair (code, spec).
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
                options.append(self._current['effect'])

            case 'SELECT' | 'TRIBUTE' | 'ANNOUNCE_RACE':
                # 'type': spec
                #
                # Return card specs to select, assume `n` is 1
                # >>> ['h3', 'h4', 's5', ... ]
                options = self._current['options']
                targets = {
                    card[1] : card[0] for card in self._current['targets'] if card[0] not in self._queue
                }

                return (options, targets)

            case _:
                raise ValueError(f"Unknown requirement: {self._current.get('requirement', None)}")

        return (options, targets)

    def last_option(self) -> Action | None:
        match self._current.get('requirement', None):
            case 'BATTLE_ACTION' | 'IDLE_ACTION':
                return self._queue[-1]
            case _:
                return None

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
                self._queue.append(action)
                return True

            case 'IDLE_ACTION':
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

            case 'SELECT' | 'TRIBUTE' | 'ANNOUNCE_RACE':
                self._queue.append(action)
                return len(self._queue) >= self._current.get('foreach', self._current['min'])

            case _:
                raise ValueError(f"Unknown requirement: {self._current.get('requirement', None)}")

    @classmethod
    def from_dict(cls, state_dict: Dict | None) -> 'StateMachine':
        return cls(state_dict) if state_dict is not None else None
