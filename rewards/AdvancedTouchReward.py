from rlgym_sim.utils.common_values import *
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np

class AdvancedTouchReward(RewardFunction):
    def __init__(self, touch_reward=1.0, acceleration_reward=0.5, use_touch_count=True):
        super().__init__()
        self.touch_reward = touch_reward
        self.acceleration_reward = acceleration_reward
        self.use_touch_count = use_touch_count
        self.prev_ball_vel = None

    def reset(self, state: GameState):
        self.prev_ball_vel = state.ball.linear_velocity

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        reward = 0
        if player.ball_touched:
            touches = 1 if not self.use_touch_count else 1  # rlgym_sim heeft geen aparte touch count
            reward += self.touch_reward * touches

            ball_vel = state.ball.linear_velocity
            prev_vel = self.prev_ball_vel
            acceleration = np.linalg.norm(ball_vel - prev_vel) / 6000  # normalized
            reward += self.acceleration_reward * acceleration

            self.prev_ball_vel = ball_vel
        return reward