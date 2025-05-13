from rlgym_sim.utils.common_values import *
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np

class FlipResetReward(RewardFunction):
    def __init__(self, heightScaling=0.25, minimumHeight=100):
        super().__init__()
        self.heightScaling = heightScaling
        self.minimumHeight = minimumHeight
        self.prevFlip = {}

    def reset(self, initial_state: GameState):
        # Reset de flip status voor elke speler
        self.prevFlip = {
            player.car_id: player.has_flip
            for player in initial_state.players
        }

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0

        # Haal de vorige flip status van deze speler op
        prev = self.prevFlip.get(player.car_id, False)

        # Als speler net een flip heeft gekregen en hoog genoeg is
        if not prev and player.has_flip and player.car_data.position[2] > self.minimumHeight:
            reward += player.car_data.position[2] * self.heightScaling

        # Update de flip status van deze speler
        self.prevFlip[player.car_id] = player.has_flip

        # Removed clipping for better learning range
        return reward


# start of opti's rewards
