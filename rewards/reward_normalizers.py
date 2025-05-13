from rlgym_sim.utils.common_values import *
from rlgym_sim.utils.reward_functions import RewardFunction
import numpy as np
import inspect

class AutoRewardNormalizer(RewardFunction):
    def __init__(self, reward_fn: RewardFunction, epsilon=1e-8):
        super().__init__()
        self.reward_fn = reward_fn
        self.epsilon = epsilon
        self.running_mean = 0
        self.running_var = 0
        self.count = 1e-4

    def reset(self, initial_state, shared_info=None):
        if shared_info is not None:
            self.reward_fn.reset(initial_state, shared_info)
        else:
            self.reward_fn.reset(initial_state)

    def get_reward(self, player, state, previous_action, shared_info=None) -> float:
        # Check of de onderliggende reward een shared_info parameter accepteert
        args = inspect.signature(self.reward_fn.get_reward).parameters
        if "shared_info" in args:
            reward = self.reward_fn.get_reward(player, state, previous_action, shared_info)
        else:
            reward = self.reward_fn.get_reward(player, state, previous_action)

        # Normalisatie
        self.count += 1
        delta = reward - self.running_mean
        self.running_mean += delta / self.count
        self.running_var += delta * (reward - self.running_mean)
        std = np.sqrt(self.running_var / self.count + self.epsilon)
        return (reward - self.running_mean) / std
