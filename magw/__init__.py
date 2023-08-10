from gymnasium.envs.registration import register

register(
     id="magw/GridWorldEnv-v0",
     entry_point="magw.envs:GridWorldEnv",
     max_episode_steps=300
)