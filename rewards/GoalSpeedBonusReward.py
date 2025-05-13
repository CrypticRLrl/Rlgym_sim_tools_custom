from numpy.linalg import norm
from rlgym_sim.utils.common_values import *
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np

class GoalSpeedBonusReward(RewardFunction):
    def __init__(self, goal_speed_bonus_w=1.0):
        super().__init__()
        self.goal_speed_bonus_w = goal_speed_bonus_w
        self.last_ball_velocity = np.zeros(3)
        self.last_blue_score = 0
        self.last_orange_score = 0

    def reset(self, initial_state: GameState):
        self.last_ball_velocity = initial_state.ball.linear_velocity
        self.last_blue_score = initial_state.blue_score
        self.last_orange_score = initial_state.orange_score

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        d_blue = state.blue_score - self.last_blue_score
        d_orange = state.orange_score - self.last_orange_score
        ball_velocity = state.ball.linear_velocity
        self.last_blue_score = state.blue_score
        self.last_orange_score = state.orange_score

        reward = 0.0
        if d_blue > 0 or d_orange > 0:
            goal_speed = 0.0
            if d_blue > 0:
                goal_speed = d_blue * norm(self.last_ball_velocity)
            elif d_orange > 0:
                goal_speed = d_orange * norm(self.last_ball_velocity)

            bonus = self.goal_speed_bonus_w * (goal_speed / BALL_MAX_SPEED)
            if player.team_num == ORANGE_TEAM and d_orange > 0:
                reward += bonus
            elif player.team_num != ORANGE_TEAM and d_blue > 0:
                reward += bonus
            elif player.team_num == ORANGE_TEAM and d_blue > 0:
                reward -= bonus
            elif player.team_num != ORANGE_TEAM and d_orange > 0:
                reward -= bonus

        self.last_ball_velocity = ball_velocity
        return float(reward)
