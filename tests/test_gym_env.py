import gymnasium as gym
from gymnasium.envs.registration import register
from pprint import pprint
from tqdm import tqdm

from env import single_gym_env as gym_env
from env.single_gym_env import YGOEnv

env = YGOEnv()

def test_to_vector():
    actions = ['h2', 'h4']
    state = {
        'hand': [8131171, 5318639, 79575620, 56120475, 88240808],
        'phase': 4,
        'remain_normal_summon': 1,
        'score': {'opponent': {'deck': 35,
                                'grave': [],
                                'hand': 5,
                                'lp': 8000,
                                'removed': []},
                    'player': {'deck': 35,
                            'grave': [],
                            'hand': 5,
                            'lp': 8000,
                            'removed': []}},
        'table': {
            'opponent': {
                'monster': [[None, 0], [None, 0], [None, 0], [None, 0], [None, 0]],
                'spell': [[None, 0], [None, 0], [None, 0], [None, 0], [None, 0]]
            },
            'player': {
                'monster': [[None, 0], [None, 0], [None, 0], [None, 0], [None, 0]],
                'spell': [[None, 0], [None, 0], [None, 0], [None, 0], [None, 0]]
            }
        }
    }

    embed = {
        '?': {
            'activate': [],
            'mset': ['h1', 'h4'],
            'repos': [],
            'requirement': 'IDLE',
            'set': ['h3', 'h5'],
            'spsummon': [],
            'summonable': [],
            'to_bp': True,
            'to_ep': True
        },
        'state': {
            'hand': [31560081, 77585513, 44095762, 79575620, 60082869],
            'phase': 4,
            'remain_normal_summon': 1,
            'score': {'opponent': {'deck': 35,
                                    'grave': [],
                                    'hand': 5,
                                    'lp': 8000,
                                    'removed': []},
                        'player': {'deck': 34,
                                'grave': [73915051],
                                'hand': 5,
                                'lp': 8000,
                                'removed': []}},
            'table': {
                'opponent': {
                    'monster': [[None, 0], [None, 0], [None, 0], [None, 0], [None, 0]],
                    'spell': [[None, 0], [None, 0], [None, 0], [None, 0], [None, 0]]
                },
                'player': {
                    'monster': [[73915052, 4], [73915053, 4], [73915054, 4], [73915055, 4], [None, 0]],
                    'spell': [[None, 0], [None, 0], [None, 0], [None, 0], [None, 0]]
                }
            }
        }
    }

    pprint(env._spec_to_card_id(embed['state'], ['h1', 'h3', 'h4', 'h5']))

    # env._encode_state(state, actions)
    # pprint(state)
    # print(env._state)
