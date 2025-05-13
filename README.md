# Rlgym_sim_tools_custom
i found out that rlgym tools worked with rlgym only and not with rlgym_sim (v2) here are my files and description on how to use them


*  replace the common_values.py in your Rlgym_sim folder with the on i made or add the missing values by hand.
*  put the reward_config.py in the same folder as you train your agent in.
*  lace the rewards folder in the same folder
*  use in your trainer file reward_fn = get_reward_fn() in your build_rocketsim_env

now you should be good to go and you can change the weights from the reward_config.py

this way in my opinion you can keep your custom rewards nice and clear.
if you add your own custom files make sure to add them to the __init__ in the rewards folder and add the import in reward_config.py
