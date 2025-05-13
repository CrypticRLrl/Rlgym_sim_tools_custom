from rlgym_sim.utils.common_values import *
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np

class SpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        car_speed = np.linalg.norm(player.car_data.linear_velocity)
        car_dir = np.sign(player.car_data.forward().dot(player.car_data.linear_velocity))
        
        if car_dir < 0:
            return 0  # Achteruit rijden krijgt geen reward
        
        reward = car_speed / CAR_MAX_SPEED
        
        # Removed clipping for better learning range
        return reward
