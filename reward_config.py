from rewards import (
    InAirReward, SpeedTowardBallReward, BallTouchReward, FaceBallReward,
    LandingRecoveryReward, SpeedReward, AerialNavigation, BoostReward,
    BoostPickupReward, FlipResetReward, AlignmentReward, BoostLoseReward,
    GoalSpeedBonusReward, AerialDistanceReward, WavedashReward,
    AdvancedTouchReward, BallTravelReward
)
from rlgym_sim.utils.reward_functions.combined_reward import CombinedReward
from rewards.reward_normalizers import AutoRewardNormalizer
from rlgym_sim.utils.reward_functions.common_rewards import EventReward

def get_reward_fn():
    """
    Returns the full reward function with AutoRewardNormalizer.
    Adjust weights below to tweak influence of each reward.
    """
    base_reward = CombinedReward.from_zipped(
        (InAirReward(), 0.1),
        (SpeedTowardBallReward(), 1.0),
        (BallTouchReward(False), 1.0),
        (FaceBallReward(), 0.6),
        (LandingRecoveryReward(), 0.5),
        (SpeedReward(), 0.2),
        (AerialNavigation(), 0.01),
        (BoostReward(), 0.2),
        (BoostPickupReward(), 0.6),
        (FlipResetReward(), 0.0),
        (AlignmentReward(), 0.3),
        (BoostLoseReward(), -0.2),
        (GoalSpeedBonusReward(), 0.5),
        (AerialDistanceReward(), 0.05),
        (WavedashReward(), 0.5),
        (AdvancedTouchReward(), 1.0),
        (BallTravelReward(), 0.7),
        (EventReward(team_goal=50, concede=-50, demo=0.1), 10.0)
    )

    return AutoRewardNormalizer(base_reward)