
import gym
import numpy as np
from mbt_gym.gym.index_names import INVENTORY_INDEX, TIME_INDEX

class ModifiedReduceStateSizeWrapper(gym.Wrapper):
    def __init__(self, env, list_of_state_indices: list = [INVENTORY_INDEX, TIME_INDEX]):
        super(ModifiedReduceStateSizeWrapper, self).__init__(env)
        assert type(env.observation_space) == gym.spaces.box.Box
        self.observation_space = gym.spaces.box.Box(
            low=env.observation_space.low[list_of_state_indices],
            high=env.observation_space.high[list_of_state_indices],
            dtype=np.float64,
        )
        self.list_of_state_indices = list_of_state_indices
        
        # Instance variable to store the original observations
        self.original_obs = None

    def reset(self):
        obs = self.env.reset()
        # Store the original observations
        self.original_obs = obs.copy()
        return obs[:, self.list_of_state_indices]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Store the original observations
        self.original_obs = obs.copy()
        return obs[:, self.list_of_state_indices], reward, done, info

    @property
    def spec(self):
        return self.env.spec
