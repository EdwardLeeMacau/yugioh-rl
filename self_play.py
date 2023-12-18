import json
import os
import shutil
import warnings
from datetime import datetime

import gymnasium as gym
from gymnasium.envs.registration import register
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env_config import ENV_CONFIG
from eval import evaluate, play_game_multiple_times
from model import MultiFeaturesExtractor
from policy import PseudoSelfPlayPolicy

warnings.filterwarnings("ignore")
register(
    id="single_ygo",
    entry_point="env.single_gym_env:YGOEnv",
    kwargs=ENV_CONFIG,
)


# Set hyper params (configurations) for training
# Modify run_id to current time
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
CONFIG = {
    "run_id": RUN_ID,

    "policy_network": "MultiInputPolicy",
    "save_path": "models",

    "epoch_num": 100,
    "update_opponent_freq": 1,
    "timesteps_per_epoch": 32768,
    "n_steps": 256,
    "clip_range": 0.2,
    "batch_size": 512,
    "parallel": 16,
    "eval_episode_num": 128,
    "learning_rate": 0.0003,
    "gamma": 0.99,
}

def make_env():
    env = gym.make('single_ygo')
    return env

def train(model, config, eval_env):
    savedir = os.path.join(config['save_path'], config['run_id'])
    with open(os.path.join(savedir, f"metrics.json"), 'w') as f:
        N = config['parallel'] // 2
        for epoch in range(1, config["epoch_num"] + 1):
            ### Train agent using SB3
            model.learn(
                total_timesteps=config["timesteps_per_epoch"],
                reset_num_timesteps=False,
                progress_bar=True,
                log_interval=1,
            )

            ### Evaluation time and save the model with higher winning rate
            trajectories = play_game_multiple_times(config['eval_episode_num'], eval_env, model)
            metrics = evaluate(trajectories)

            ### Evaluation
            print(f'{config["run_id"]}. Epoch: {epoch}')
            print(metrics.describe())

            f.write(metrics.mean().to_json() + '\n')
            f.flush()

            # Always save the model
            model.save(os.path.join(savedir, str(epoch)))

            # Make opponent diverse
            if epoch % config['update_opponent_freq'] == 0:
                policy = PseudoSelfPlayPolicy(
                    os.path.join('models', RUN_ID, f'{epoch-(0 * config["update_opponent_freq"])}.zip')
                )
                for env in model.env.unwrapped.envs[N:N+int(N/2)]:
                    env.opponent = policy

                if epoch - 1 * config['update_opponent_freq'] <= 0:
                    continue

                policy = PseudoSelfPlayPolicy(
                    os.path.join('models', RUN_ID, f'{epoch-(1 * config["update_opponent_freq"])}.zip')
                )
                for env in model.env.unwrapped.envs[N+int(N/2):N+int(N/2)+int(N/4)]:
                    env.opponent = policy

                if epoch - 2 * config['update_opponent_freq'] <= 0:
                    continue

                policy = PseudoSelfPlayPolicy(
                    os.path.join('models', RUN_ID, f'{epoch-(2 * config["update_opponent_freq"])}.zip')
                )
                for env in model.env.unwrapped.envs[N+int(N/2)+int(N/4)+int(N/8):]:
                    env.opponent = policy


if __name__ == "__main__":
    train_env = DummyVecEnv([make_env for _ in range(CONFIG["parallel"])])
    eval_env = make_env()
    model = MaskablePPO(
        CONFIG["policy_network"],
        train_env,
        learning_rate=CONFIG["learning_rate"],
        gamma=CONFIG["gamma"],
        n_steps=CONFIG["n_steps"],
        clip_range=CONFIG['clip_range'],
        batch_size=CONFIG['batch_size'],
        tensorboard_log=os.path.join("logs", CONFIG["run_id"]),
        policy_kwargs={
            "features_extractor_class": MultiFeaturesExtractor,
            "features_extractor_kwargs":{},
        }
    )

    # Backup config.py to the script output directory
    os.makedirs(os.path.join(CONFIG['save_path'], CONFIG["run_id"]), exist_ok=True)
    shutil.copyfile(
        "env_config.py",
        os.path.join(CONFIG['save_path'], CONFIG["run_id"], "env_config.py")
    )
    with open(os.path.join(CONFIG['save_path'], CONFIG['run_id'], "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)

    train(model, CONFIG, eval_env)
    train_env.close()
    eval_env.close()
