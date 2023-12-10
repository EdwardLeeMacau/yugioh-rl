import os
import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv

from model import MultiFeaturesExtractor
from env.single_gym_env import YGOEnv

from eval import play_game_multiple_times, calc_winning_rate, evaluate

warnings.filterwarnings("ignore")
register(
    id="single_ygo",
    entry_point="env.single_gym_env:YGOEnv",
    kwargs={
        'advantages': {
            'player1': { 'lifepoints': 8000 },
            'player2': { 'lifepoints': 8000 }
        },
    }
    # TODO: Parse the arguments from the config file
)


# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",

    "policy_network": "MultiInputPolicy",
    "save_path": "models/sample_model",

    "epoch_num": 100,
    "timesteps_per_epoch": 4096,
    "n_steps": 128,
    "parallel": 32,
    "eval_episode_num": 100,
}

def make_env():
    env = gym.make('single_ygo')
    return env

def train(model, config, eval_env):
    current_best_ = 0
    current_best = 0

    max_winning_rate = 0.0

    for epoch in range(config["epoch_num"]):
        ### Train agent using SB3
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            progress_bar=True,
            log_interval=1,
        )

        ### Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)

        ### Evaluation time and save the model with higher winning rate
        trajectories = play_game_multiple_times(config['eval_episode_num'], eval_env, model)
        metrics = evaluate(trajectories)
        print(f"Winning rate: {metrics['winning_rate']:.2%}")

        model.save(f"{config['save_path']}/{epoch}")
        print("---------------")

    # Workaround for terminating the background threads
    ...


if __name__ == "__main__":
    train_env = DummyVecEnv([make_env for _ in range(my_config["parallel"])])
    eval_env = make_env()
    model = MaskablePPO(
        my_config["policy_network"],
        train_env,
        learning_rate=0.0003,
        gamma=0.90,
        n_steps=my_config["n_steps"],
        tensorboard_log=os.path.join("logs", my_config["run_id"]),
        policy_kwargs={
            "features_extractor_class": MultiFeaturesExtractor,
            "features_extractor_kwargs":{},
        }
    )
    train(model, my_config, eval_env)
