import torch

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn, Tensor

from env.single_gym_env import YGOEnv

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MultiFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self,
            observation_space: spaces.Dict,
            feature_dim: int = 256,
        ):
        super().__init__(observation_space, feature_dim)

        # ! Use obs.n if obs is a Discrete space
        # ! Use obs.shape[0] if obs is a Box space
        # ! Use obs.nvec.sum() if obs is a MultiDiscrete space
        self.LP_FE            = MLP(2 * observation_space['agent_LP'].shape[0],
                                    feature_dim, feature_dim // 2)
        self.action_mask_FE   = MLP(observation_space['action_mask'].shape[0],
                                    feature_dim, feature_dim // 2)
        self.phase_FE         = MLP(observation_space['phase'].n,
                                    feature_dim, feature_dim // 2)
        self.agent_deck_FE    = MLP(observation_space['agent_deck'].n,
                                    feature_dim, feature_dim // 2)
        self.oppo_deck_FE     = MLP(observation_space['oppo_deck'].n,
                                    feature_dim, feature_dim // 2)
        self.oppo_hand_FE     = MLP(observation_space['oppo_hand'].n,
                                    feature_dim, feature_dim // 2)
        self.agent_hand_FE    = MLP(observation_space['agent_hand'].shape[0],
                                    feature_dim, feature_dim // 2)
        self.agent_grave_FE   = MLP(observation_space['agent_grave'].shape[0],
                                    feature_dim, feature_dim // 2)
        self.agent_removed_FE = MLP(observation_space['agent_removed'].shape[0],
                                    feature_dim, feature_dim // 2)
        self.oppo_grave_FE    = MLP(observation_space['oppo_grave'].shape[0],
                                    feature_dim, feature_dim // 2)
        self.oppo_removed_FE  = MLP(observation_space['oppo_removed'].shape[0],
                                    feature_dim, feature_dim // 2)
        self.post_agent_m_FE  = MLP(observation_space['t_agent_m'].n,
                                    feature_dim, feature_dim // 2)
        self.post_oppo_m_FE   = MLP(observation_space['t_oppo_m'].n,
                                    feature_dim, feature_dim // 2)
        self.post_agent_s_FE  = MLP(observation_space['t_agent_m'].n,
                                    feature_dim, feature_dim // 2)
        self.post_oppo_s_FE   = MLP(observation_space['t_agent_m'].n,
                                    feature_dim, feature_dim // 2)

    def forward(self, observations) -> Tensor:
        single_int_input = torch.cat([observations['agent_LP'], observations['oppo_LP']], dim=-1)

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
