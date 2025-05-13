from rlgym_sim.utils.common_values import *
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np

class BoostPickupReward(RewardFunction):
    # Constructor to initialize prevBoost
    def __init__(self):
        super().__init__()
        self.prevBoost = 0  # Store previous boost amount as an instance variable

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        self.prevBoost = 0  # Reset previous boost amount on reset

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0

        # If the player's boost has increased, reward the difference
        if player.boost_amount > self.prevBoost:
            reward = (player.boost_amount - self.prevBoost) * 0.2

        # Update prevBoost for the next step
        self.prevBoost = player.boost_amount

        # Removed clipping for better learning range
        return reward
