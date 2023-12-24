from policy import RandomPolicy, PseudoSelfPlayPolicy

ENV_CONFIG = {
    "opponent": RandomPolicy(),
    # "opponent": PseudoSelfPlayPolicy(model_path='models/0.zip'),
    'advantages': {
        'player1': { 'lifepoints': 8000 },
        'player2': { 'lifepoints': 8000 }
    },

    # 'reward_kwargs': {
    #     'type': 'win/loss',
    # },

    # 'reward_kwargs': {
    #     'type': 'step_penalty',
    #     'weight': 0.005
    # },

    # 'reward_kwargs': {
    #     'type': 'step_reward',
    #     'weight': 1.0
    # },

    # 'reward_kwargs': {
    #     'type': 'LP',
    #     'weight': 0.05
    # },

    # 'reward_kwargs': {
    #     'type': 'LP_linear_step',
    #     'weight': 0.1
    # },

    'reward_kwargs': {
        'type': 'LP_exp_step',
        'temperature': 10.0,
        'weight': 0.5
    },
}
