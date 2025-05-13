from rlgym_sim.utils.common_values import *
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
import numpy as np

class FlipResetRewardOPTI(RewardFunction):
    def __init__(self):
        super().__init__()
        # Reward weights and parameters (set these based on your needs)
        self.flip_reset_w = 10  # Reward for a successful flip reset
        self.quick_flip_reset_w = 5  # Reward for quick first flip reset
        self.quick_flip_reset_norm_steps = 100  # Normalization steps for quick flip reset
        self.flip_reset_delay_steps = 50  # Delay steps for subsequent resets
        self.inc_flip_reset_w = 2  # Incremental reward for consecutive resets
        self.prevent_chain_reset = True  # Whether to prevent consecutive resets
        self.cancel_flip_reset_indices = None  # Actions to cancel a flip reset (can be filled with a list of indices)

        self.got_reset = [False] * 8  # Keep track of reset status for each player (assuming 8 players)
        self.cons_resets = 0  # Counter for consecutive resets
        self.reset_timer = -100000  # Timer to track time since last reset
        self.kickoff_timer = 1000  # Set the time until the kickoff is finished

    def reset(self, initial_state: GameState):
        # Reset logic when the game resets (this is called at the start of the game)
        self.got_reset = [False] * 8  # Reset reset status for each player
        self.cons_resets = 0  # Reset the consecutive reset counter
        self.reset_timer = -100000  # Reset the reset timer

    def get_reward(self, player: PlayerData, state: GameState) -> float:
        reward = 0

        # Loop over all players (assuming `player` is the current player whose reward is being calculated)
        i = player.index  # Get the player's index
        last = state.players[i - 1]  # The last player's data (for checking the jump state)

        # Check for flip reset (first flip reset of the episode)
        if not last.has_jump and player.has_jump and state.ball.position[2] > 200 and \
                np.linalg.norm(state.ball.position - player.car_data.position) < 110 and \
                cosine_similarity(state.ball.position - player.car_data.position, -player.car_data.up()) > 0.9:
            if not self.got_reset[i]:  # First reset of episode
                reward += self.quick_flip_reset_w * self.quick_flip_reset_norm_steps / self.kickoff_timer
            self.got_reset[i] = True

            # Reward for successful flip reset (after first reset)
            if (self.kickoff_timer - self.reset_timer > self.flip_reset_delay_steps and self.prevent_chain_reset) or \
                    not self.prevent_chain_reset:
                if previous_action is not None and self.cancel_flip_reset_indices is not None and \
                        previous_action[i] not in self.cancel_flip_reset_indices:
                    reward += self.flip_reset_w
                self.cons_resets += 1
                if self.cons_resets > 1:
                    reward += self.inc_flip_reset_w * min((1.4 ** self.cons_resets), 6) / 6
            self.reset_timer = self.kickoff_timer

        # Reset the counters if the player is on the ground (no more flip resets allowed)
        elif player.on_ground:
            self.cons_resets = 0
            self.reset_timer = -100000

        # Removed clipping for better learning range
        return reward
