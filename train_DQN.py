import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
import torch as th
import torch.nn as nn

#import wandb
#from wandb.integration.sb3 import WandbCallback

import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

warnings.filterwarnings("ignore")
register(
    id="single_ygo",
    entry_point="env.single_gym_env:YGOEnv"
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",

    "algorithm": DQN,
    "policy_network": "MultiInputPolicy",
    "save_path": "models/sample_model",

    "epoch_num": 100,
    "timesteps_per_epoch": 25000,
    "eval_episode_num": 10,

    #"normalize_images": False,
}

def make_env():
    env = gym.make('single_ygo')
    return env

def train(env, model, config):
    current_best_ = 0
    current_best = 0

    for epoch in range(config["epoch_num"]):

        ### Train agent using SB3
        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
        )

        ### Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_score = 0
        avg_highest = 0
        for seed in range(config["eval_episode_num"]):
            done = False

            # Set seed using old Gym API
            env.seed(seed)
            obs = env.reset()

            # Interact with env using old Gym API
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

        ### Save best model
        if epoch % 10 == 0:
            print("Saving Model")
            save_path = config["save_path"]
            model.save(f"{save_path}/{epoch}")
        print("---------------")

class MultiFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self,
                observation_space: spaces.Dict,
                feature_dim: int = 256,
                ):
        super().__init__(observation_space, feature_dim)
        self.LP_FE = nn.Sequential(
            nn.Linear(2, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim)
        )
        self.phase_FE = nn.Sequential(
            nn.Linear(6, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim)
        )
        self.agent_deck_FE = nn.Sequential(
            nn.Linear(41, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim)
        )
        self.oppo_deck_FE = nn.Sequential(
            nn.Linear(41, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim)
        )
        self.oppo_hand_FE = nn.Sequential(
            nn.Linear(7, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim)
        )
        self.agent_hand_FE = nn.Sequential(
            nn.Linear(160, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim),
        )
        self.agent_grave_FE = nn.Sequential(
            nn.Linear(160, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim),
        )
        self.agent_removed_FE = nn.Sequential(
            nn.Linear(160, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim),
        )
        self.oppo_grave_FE = nn.Sequential(
            nn.Linear(160, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim),
        )
        self.oppo_removed_FE = nn.Sequential(
            nn.Linear(160, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim),
        )
        self.post_agent_m_FE = nn.Sequential(
            nn.Linear(225, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim),
        )
        self.post_oppo_m_FE = nn.Sequential(
            nn.Linear(225, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim),
        )
        self.post_agent_s_FE = nn.Sequential(
            nn.Linear(225, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim),
        )
        self.post_oppo_s_FE = nn.Sequential(
            nn.Linear(225, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim),
        )

    def forward(self, observations) -> th.Tensor:
        single_int_input = th.cat([observations['agent_LP'], observations['oppo_LP']], dim=-1)
        embedded_LP = self.LP_FE(single_int_input)

        embedded_agent_hand = self.agent_hand_FE(observations['agent_hand'])
        embedded_agent_grave = self.agent_grave_FE(observations['agent_grave'])
        embedded_agent_removed = self.agent_removed_FE(observations['agent_removed'])
        embedded_oppo_grave = self.oppo_grave_FE(observations['oppo_grave'])
        embedded_oppo_removed = self.oppo_removed_FE(observations['oppo_removed'])

        embedded_agent_m = self.post_agent_m_FE(observations['t_agent_m'])
        embedded_oppo_m = self.post_oppo_m_FE(observations['t_oppo_m'])
        embedded_agent_s = self.post_agent_s_FE(observations['t_agent_s'])
        embedded_oppo_s = self.post_oppo_s_FE(observations['t_oppo_s'])

        embedded_phase = self.phase_FE(observations['phase'])
        embedded_agent_deck = self.agent_deck_FE(observations['agent_deck'])
        embedded_oppo_deck = self.oppo_deck_FE(observations['oppo_deck'])
        embedded_oppo_hand = self.oppo_hand_FE(observations['oppo_hand'])

        FE_output = embedded_agent_hand + embedded_agent_grave + embedded_agent_removed + embedded_oppo_grave + embedded_oppo_removed + embedded_agent_m + embedded_agent_s + embedded_oppo_m + embedded_oppo_s + embedded_phase + embedded_agent_deck + embedded_oppo_deck + embedded_oppo_hand

        return FE_output


if __name__ == "__main__":
    env = DummyVecEnv([make_env for _ in range(1)])
    model = my_config["algorithm"](
        my_config["policy_network"],
        env,
        learning_rate=0.0007,#0.00007,
        gamma=0.099,
        verbose=0,
        seed=1109,
        buffer_size=10,
        #tensorboard_log=my_config["run_id"],
        policy_kwargs={
            "features_extractor_class": MultiFeaturesExtractor,
            "features_extractor_kwargs":{},
        }
    )
    breakpoint()
    train(env, model, my_config)
