from rlgym_sim.utils.common_values import CAR_MAX_SPEED
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np


class WavedashReward(RewardFunction):
    def __init__(self, scale_by_acceleration: bool = True):
        super().__init__()
        self.scale_by_acceleration = scale_by_acceleration
        self.prev_state = None
        self.prev_acceleration = {}

    def reset(self, initial_state: GameState, shared_info=None):
        self.prev_state = initial_state
        self.prev_acceleration = {p.car_id: 0 for p in initial_state.players}

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        car_id = player.car_id
        prev_player = next((p for p in self.prev_state.players if p.car_id == car_id), None)
        if prev_player is None:
            return 0

        # Gebruik player.on_ground en player.has_flip (correct attribuut)
        wavedash = player.on_ground and not prev_player.on_ground and (player.has_flip or prev_player.has_flip)

        if self.scale_by_acceleration:
            if player.has_flip and not prev_player.has_flip:
                acc = np.linalg.norm(player.car_data.linear_velocity - prev_player.car_data.linear_velocity)
                self.prev_acceleration[car_id] = acc
            if wavedash:
                acc = self.prev_acceleration[car_id]
                self.prev_acceleration[car_id] = 0
                return acc / CAR_MAX_SPEED
            elif not player.has_flip:
                self.prev_acceleration[car_id] = 0
            return 0
        else:
            return 1 if wavedash else 0
