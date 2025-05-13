from rlgym_sim.utils.common_values import *
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np

class BoostLoseReward(RewardFunction):
    def __init__(self, boost_lose_w=1.0):
        super().__init__()
        self.boost_lose_w = boost_lose_w
        self.last_boost_amount = {}

    def reset(self, initial_state: GameState):
        self.last_boost_amount = {
            player.car_id: player.boost_amount
            for player in initial_state.players
        }

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        last_boost = self.last_boost_amount.get(player.car_id, player.boost_amount)
        current_boost = player.boost_amount
        boost_diff = np.sqrt(np.clip(current_boost, 0, 1)) - np.sqrt(np.clip(last_boost, 0, 1))
        self.last_boost_amount[player.car_id] = current_boost

        reward = 0.0
        if boost_diff < 0:
            car_height = player.car_data.position[2]
            penalty = self.boost_lose_w * boost_diff * (1 - car_height / 642.775) #goal height
            reward += penalty
        return float(reward)
