from rlgym_sim.utils.common_values import *
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np

class BallTravelReward(RewardFunction):
    def __init__(self, weight=1.0, normalization_factor=1/8192):
        super().__init__()
        self.weight = weight
        self.normalization_factor = normalization_factor
        self.prev_ball_pos = None
        self.last_touch_id = None
        self.distance_since_touch = 0

    def reset(self, state: GameState):
        self.prev_ball_pos = state.ball.position
        self.last_touch_id = None
        self.distance_since_touch = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        ball_pos = state.ball.position
        car_id = player.car_id

        dist = np.linalg.norm(ball_pos - self.prev_ball_pos)
        self.prev_ball_pos = ball_pos
        self.distance_since_touch += dist

        reward = 0
        if player.ball_touched:
            norm_dist = self.distance_since_touch * self.normalization_factor
            if car_id == self.last_touch_id:
                reward = norm_dist * self.weight
            else:
                reward = norm_dist * self.weight
            self.last_touch_id = car_id
            self.distance_since_touch = 0

        return reward