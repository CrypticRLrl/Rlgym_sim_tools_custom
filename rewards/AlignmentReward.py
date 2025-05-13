from rlgym_sim.utils.math import cosine_similarity
from rlgym_sim.utils.common_values import *
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np

class AlignmentReward(RewardFunction):
    def __init__(self, align_w=1.0):
        super().__init__()
        self.align_w = align_w

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_pos = state.ball.position
        player_pos = player.car_data.position

        if player.team_num == ORANGE_TEAM:
            goal_vector = np.array(ORANGE_GOAL_BACK) - player_pos
        else:
            goal_vector = np.array(BLUE_GOAL_BACK) - player_pos

        player_to_ball = ball_pos - player_pos
        alignment = cosine_similarity(player_to_ball, goal_vector)

        reward = self.align_w * alignment
        return float(reward)
