import math
import random

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=16, nagents=16):
        """
        """
        assert size > 2
        assert nagents > 0
        assert nagents < (0.25 * size * size), "There cannot be more nagents than 25 percent of total grid squares"

        self.size = size  # The size of the square grid
        self.nagents = nagents
        self.window_size = 720  # The size of the PyGame window

        self.observation_space = spaces.Dict({
                "locations": spaces.Box(0, size - 1, shape=(nagents,2), dtype=int),
                "targets": spaces.Box(0, size - 1, shape=(nagents,2), dtype=int)
            }
        )

        # We have 4 actions, corresponding to "stop", "right", "up", "left" and "down"
        self.action_space = spaces.Dict({
                "agent_to_move": spaces.Discrete(nagents),
                "action": spaces.Discrete(5)
        })

        self._action_to_direction = [
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([0, -1])
        ]

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _total_distance(self):
        return np.sum(np.abs(self._agent_locations - self._agent_targets))

    def _get_obv(self):
        return {
            "locations": self._agent_locations,
            "targets": self._agent_targets
        }
    
    def _get_info(self):
        return {
            "grid": self._grid,
            "size": self.size,
            "total": self._previous_total_distance
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # https://stackoverflow.com/questions/55244113/python-get-random-unique-n-pairs
        def decode(i):
            k = math.floor((1+math.sqrt(1+8*i))/2)
            if i % 2:
                return i-k*(k-1)//2,k
            else:
                return k,i-k*(k-1)//2
        def rand_pairs(n,m):
            return np.array([decode(i) for i in random.sample(range(n*(n-1)//2),m)])
        
        # Generate agents locations
        self._agent_locations = rand_pairs(self.size, self.nagents)
        self._agent_targets = rand_pairs(self.size, self.nagents)

        # 0 means that the square is free, more than 0, that the squera is occupied
        self._grid = np.full((self.size, self.size), 0, dtype=int)
        self._grid[self._agent_locations[:, 0], self._agent_locations[:, 1]] = 1

        # Initial total distance
        self._previous_total_distance = self._total_distance()

        observation = self._get_obv()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self._agent_to_move = 0

        return observation, info
    
    def step(self, action):
        agent_to_move = action["agent_to_move"]
        old = np.array(self._agent_locations[agent_to_move])
        direction = self._action_to_direction[action["action"]]

        # We use `np.clip` to make sure we don't leave the grid
        new = np.clip(
            old + direction, 0, self.size - 1
        )

        self._agent_locations[agent_to_move] = new

        # If two agents have the same location, we also terminate
        invalid_action = self._grid[new[0], new[1]] == 1

        # An episode is done if all agents have reached the target
        terminated = invalid_action or np.array_equal(self._agent_locations, self._agent_targets)

        # Update grid with new agent location
        self._grid[old[0], old[1]] = 0
        self._grid[new[0], new[1]] = 1

        # Maximun distance an agent can be from its target
        max_distance = (self.size - 1) * 2
        # Maximun total distance of all agents combined
        max_distance = self.nagents * max_distance

        # Compute distance to target of all agents
        total_distance = self._total_distance()
        reward = self._previous_total_distance - total_distance if not invalid_action else -1
        self._previous_total_distance = total_distance

        observation = self._get_obv()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Plot targets
        for target in self._agent_targets:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * target,
                    (pix_square_size, pix_square_size),
                ),
             )
            
        # Plot agents
        for location in self._agent_locations:
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (location + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )