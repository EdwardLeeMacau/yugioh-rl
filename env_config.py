from policy import RandomPolicy, PseudoSelfPlayPolicy

ENV_CONFIG = {
    "opponent": RandomPolicy(),
    # "opponent": PseudoSelfPlayPolicy(model_path='models/0.zip'),
    'advantages': {
        'player1': { 'lifepoints': 8000 },
        'player2': { 'lifepoints': 8000 }
    },
    'reward_type': 'step count reward'
    # reward type should be 'win/loss', 'LP', or 'step count reward'
}
