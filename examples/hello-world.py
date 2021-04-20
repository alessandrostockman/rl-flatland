import numpy as np
import time

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_manual_specifications_generator
from flatland.utils.rendertools import RenderTool

# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)
specs = [[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
         [(0, 0), (0, 0), (0, 0), (0, 0), (7, 0), (0, 0)],
         [(7, 270), (1, 90), (1, 90), (1, 90), (2, 90), (7, 90)],
         [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]]

env = RailEnv(width=6, height=4, rail_generator=rail_from_manual_specifications_generator(specs), number_of_agents=3)
env.reset()

env_renderer = RenderTool(env)
#env_renderer.render_env(show=True, show_predictions=False, show_observations=False)


for step in range(100):

    #_action = my_controller()
    #obs, all_rewards, done, info = env.step(_action)
    #print("Rewards: {}, [done={}]".format( all_rewards, done))
    env_renderer.render_env(show=True, frames=False, show_observations=False)
    time.sleep(0.3)

# uncomment to keep the renderer open
# input("Press Enter to continue...")