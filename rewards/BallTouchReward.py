from rlgym_sim.utils.common_values import *
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np

class BallTouchReward(RewardFunction):
    # Default constructor
    def __init__(self, Beginner: bool):
        super().__init__()
        self.isBeginner = Beginner
        self.ballVelLast = 0  # Initialize the previous ball velocity to 0

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        self.ballVelLast = 0  # Reset ball velocity on game reset

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0

        if self.isBeginner:
            if player.ball_touched:  # Use player data for ball touch check
                reward += 0.1
        else:
            if player.ball_touched:
                # Calculate velocity gain
                current_ball_velocity = np.linalg.norm(state.ball.linear_velocity)  # Magnitude of ball velocity
                velocity_gain = current_ball_velocity - self.ballVelLast
                reward += max(0, velocity_gain)  # Reward only positive velocity gain
                self.ballVelLast = current_ball_velocity  # Update ball velocity

        # Removed clipping for better learning range
        return reward
