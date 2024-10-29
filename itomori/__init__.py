from gymnasium.envs.registration import register

register(
     id="itomori/Navigation2DEnv-v0",
     entry_point="itomori.envs:Navigation2DEnv",
     max_episode_steps=300,
)

register(
     id="itomori/DiscreteNavigation2DEnv-v0",
     entry_point="itomori.envs:DiscreteNavigation2DEnv",
     max_episode_steps=300,
)