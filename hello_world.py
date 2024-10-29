import gymnasium as gym
import matplotlib.pyplot as plt

import itomori

map_filepath="itomori/envs/navigation2d_configs/maps/20x20/20x20_TwoObstacles_CorridorDetour.json"

env = gym.make(
    'itomori/Navigation2DEnv-v0',
    render_mode="rgb_array",
    map_filepath=map_filepath,
)

root_observation, info = env.reset()

img = env.render()

plt.imshow(img)
plt.waitforbuttonpress()

env.close()
