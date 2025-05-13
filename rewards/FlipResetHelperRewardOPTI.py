from rlgym_sim.utils.common_values import *
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np

class FlipResetHelperRewardOPTI(RewardFunction):
    def __init__(self, flip_reset_help_w=1.0):
        """
        Constructor to initialize the flip reset help weight.
        :param flip_reset_help_w: Weight to scale the flip reset reward (default is 1.0).
        """
        super().__init__()
        self.flip_reset_help_w = flip_reset_help_w

    def reset(self, initial_state: GameState):
        """
        Reset does nothing in this case.
        """
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """
        Calculate the reward for a specific player, at the current state.
        This reward encourages the bot to position itself for a potential flip reset.

        :param player: The PlayerData object representing the player.
        :param state: The current GameState.
        :param previous_action: The previous action taken by the player.
        :return: The calculated reward.
        """
        reward = 0

        if self.flip_reset_help_w != 0:
            # Calculate the 'upness' of the player's car (how aligned it is with the ceiling)
            upness = cosine_similarity(
                np.asarray([0, 0, CEILING_Z - player.car_data.position[2]]),
                -player.car_data.up()  # bottom of the car points towards the ceiling
            )

            # Calculate how far the player is from the walls
            from_wall_ratio = min(1, abs(state.ball.position[0]) / 1300)

            # Calculate the height of the ball relative to the field
            height_ratio = min(1, state.ball.position[2] / 1700)

            # Calculate how aligned the ball is with the bottom of the car
            bottom_ball_ratio = 2 * cosine_similarity(
                state.ball.position - player.car_data.position, -player.car_data.up()
            )

            # Determine the goal objective based on the player's team
            if player.team_num == BLUE_TEAM:
                objective = np.array(ORANGE_GOAL_BACK)
            else:
                objective = np.array(BLUE_GOAL_BACK)

            # Calculate how aligned the player is with the objective (goal)
            align_ratio = cosine_similarity(
                objective - player.car_data.position, player.car_data.forward()
            )

            # Calculate the positional difference between the player and the ball, with extra weight on the Z-axis (height)
            pos_diff = state.ball.position - player.car_data.position
            pos_diff[2] *= 2  # Make the Z-axis difference more important
            norm_pos_diff = np.linalg.norm(pos_diff)

            # Calculate the final flip reset reward based on the factors
            flip_rew = bottom_ball_ratio * from_wall_ratio * height_ratio * align_ratio * \
                       np.clip(-1, 1, 40 * upness / (norm_pos_diff + 1))

            # Apply the weight factor to the final reward
            reward += self.flip_reset_help_w * flip_rew

        # Removed clipping for better learning range
        return reward
