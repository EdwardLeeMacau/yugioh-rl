import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
import torch as th
import torch.nn as nn

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import stable_baselines3

warnings.filterwarnings("ignore")
register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",

    "algorithm": DQN,
    "policy_network": "CnnPolicy",
    "save_path": "models/sample_model",

    "epoch_num": 100,
    "timesteps_per_epoch": 25000,
    "eval_episode_num": 10,
    
    #"normalize_images": False,
}

def make_env():
    env = gym.make('2048-v0')
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
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     verbose=2,
            # ),
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
            
            avg_highest += info[0]['highest']/config["eval_episode_num"]
            avg_score   += info[0]['score']/config["eval_episode_num"]
        
        print("Avg_score:  ", avg_score)
        print("Avg_highest:", avg_highest)
        print()
        # wandb.log(
        #     {"avg_highest": avg_highest,
        #      "avg_score": avg_score}
        # )
        

        ### Save best model
        if current_best < avg_highest:
            print("Saving Model")
            current_best = avg_highest
            save_path = config["save_path"]
            #model.save(f"{save_path}/{epoch}")
            model.save(f"{save_path}/best_high")
        if current_best_ < avg_score:
            print("Saving Model Ave Score")
            current_best_ = avg_score
            save_path = config["save_path"]
            model.save(f"{save_path}/best_score")

        print("---------------")

    print("Avg_score:  ", current_best_)
    print("Avg_highest:", current_best)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    # run = wandb.init(
    #     project="assignment_3",
    #     config=my_config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     id=my_config["run_id"]
    # )

    env = DummyVecEnv([make_env])
    #env = VecVideoRecorder([make_env])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages

    #Policy_network = stable_baselines3.a2c.CnnPolicy(env.observation_space, env.action_space, None, normalize_images=False)
    model = my_config["algorithm"](
        #Policy_network,
        my_config["policy_network"], 
        env, 
        learning_rate=0.0007,#0.00007,
        gamma=0.099,
        verbose=0,
        seed=1109,
        tensorboard_log=my_config["run_id"],
        policy_kwargs={"features_extractor_class": CustomCNN},
    )
    train(env, model, my_config)
