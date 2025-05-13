import numpy as np
from rlgym_sim.utils.common_values import *
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rewards.FaceBallReward import FaceBallReward


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def distance2D(pos1, pos2):
    return np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))



class AerialNavigation(RewardFunction):
    def __init__(self, ball_height_min=400, player_height_min=200, beginner=True) -> None:
        super().__init__()
        self.ball_height_min = ball_height_min
        self.player_height_min = player_height_min
        self.face_reward = FaceBallReward()
        self.beginner = beginner
        self.previous_distance = None

    def reset(self, initial_state: GameState) -> None:
        self.face_reward.reset(initial_state)
        self.previous_distance = None

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0

        if (
            not player.on_ground
            and state.ball.position[2] > self.ball_height_min > player.car_data.position[2]
            and player.car_data.linear_velocity[2] > 0
            and distance2D(player.car_data.position, state.ball.position) < state.ball.position[2] * 3
        ):
            ball_direction = normalize(state.ball.position - player.car_data.position)
            alignment = ball_direction.dot(normalize(player.car_data.linear_velocity))
            alignment = min(max(alignment, 0), 1)  # clamp

            if self.beginner:
                reward += alignment * 0.5

            reward += alignment * (np.linalg.norm(player.car_data.linear_velocity) / CAR_MAX_SPEED)

            current_distance = distance2D(player.car_data.position, state.ball.position)
            if self.previous_distance is not None:
                distance_diff = self.previous_distance - current_distance
                if distance_diff > 0:
                    reward += 0.05
            self.previous_distance = current_distance

            if player.ball_touched:
                reward *= 1.5

        return reward
