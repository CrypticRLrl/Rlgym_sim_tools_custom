from rlgym_sim.utils.common_values import *
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np
from rlgym_sim.utils.common_values import CAR_MAX_SPEED

class SpeedTowardBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        player_vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        dist_to_ball = np.linalg.norm(pos_diff)

        if dist_to_ball == 0:
            return 0  # avoid division by zero

        dir_to_ball = pos_diff / dist_to_ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)

        if speed_toward_ball > 0:
            reward = speed_toward_ball / CAR_MAX_SPEED
            # Removed clipping for better learning range
            return reward
        else:
            return 0
