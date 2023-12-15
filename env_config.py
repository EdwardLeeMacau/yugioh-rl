from policy import RandomPolicy, PseudoSelfPlayPolicy

ENV_CONFIG = {
    "opponent": RandomPolicy(),
    # "opponent": PseudoSelfPlayPolicy(model_path='models/0.zip'),
    'advantages': {
        'player1': { 'lifepoints': 8000 },
        'player2': { 'lifepoints': 8000 }
    },

    # Remove reward shaping, only use win/loss reward
    'reward_kwargs': {
        'type': 'win/loss',
    },

    # Experiment range: [0.05, 0.1, 0.2]
    # 'reward_kwargs': {
    #     'type': 'LP',
    #     'weight': 0.05
    # },

    # Experiment range: [0.5, 1.0, 2.0]
    # 'reward_kwargs': {
    #     'type': 'LP_linear_step',
    #     'weight': 0.1
    # },

    # Experiment range: ?
    # 'reward_kwargs': {
    #     'type': 'LP_exp_step',
    #     'temperature': 0.1,
    #     'weight': 0.1
    # },
}
