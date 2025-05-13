from rlgym_sim.utils.common_values import *
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np

class AerialDistanceReward(RewardFunction):
    def __init__(self, height_scale = 0.05, distance_scale = 0.05):
        """
        Initializes the AerialDistanceReward function.
        :param height_scale: Scaling factor for rewarding height during aerial play.
        :param distance_scale: Scaling factor for rewarding distances traveled during aerial play.
        """
        super().__init__()
        self.height_scale = height_scale  # Scale for height-based rewards
        self.distance_scale = distance_scale  # Scale for distance-based rewards

        # Variables to track current player, previous state, and distances
        self.current_car: Optional[PlayerData] = None
        self.prev_state: Optional[GameState] = None
        self.ball_distance: float = 0  # Distance the ball has traveled
        self.car_distance: float = 0  # Distance the car has traveled

    def reset(self, initial_state: GameState):
        """
        Resets the state of the reward function.
        :param initial_state: The initial state of the game.
        """
        self.current_car = None  # No player currently tracked
        self.prev_state = initial_state  # Store the initial state for comparison

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """
        Calculates the aerial distance reward for the given player.
        :param player: The current player.
        :param state: The current game state.
        :param previous_action: The last action taken by the player (not used here).
        :return: A scaled reward value for aerial distance and touches.
        """
        rew = 0  # Initialize reward
        is_current = self.current_car is not None and self.current_car.car_id == player.car_id  # Check if player is tracked

        # Check if the player is on the ground
        if player.car_data.position[2] < RAMP_HEIGHT:
            if is_current:  # If the tracked player is now on the ground, reset tracking
                is_current = False
                self.current_car = None
        # Detect the first aerial touch
        elif player.ball_touched and not is_current:
            is_current = True  # Start tracking the player
            self.ball_distance = 0  # Reset ball distance
            self.car_distance = 0  # Reset car distance
            # Reward for initial aerial height, scaled by height_scale
            rew = self.height_scale * max(player.car_data.position[2] + state.ball.position[2] - 2 * RAMP_HEIGHT, 0)
        # If the player is still in the air after the initial touch
        elif is_current:
            # Accumulate car travel distance since the last frame
            self.car_distance += np.linalg.norm(player.car_data.position - self.current_car.car_data.position)
            # Accumulate ball travel distance since the last frame
            self.ball_distance += np.linalg.norm(state.ball.position - self.prev_state.ball.position)
            # Reward for additional touches, based on accumulated distances
            if player.ball_touched:
                rew = self.distance_scale * (self.car_distance + self.ball_distance)  # Reward based on total distance
                self.car_distance = 0  # Reset car distance
                self.ball_distance = 0  # Reset ball distance

        if is_current:
            # Update current car to the latest player data for tracking
            self.current_car = player

        # Update previous state for the next frame comparison
        self.prev_state = state

        # Normalize reward by the maximum possible distance (2 * BACK_WALL_Y)
        # Removed clipping for better learning range
        return rew / (2 * BACK_WALL_Y)
