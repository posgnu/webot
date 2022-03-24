import os
from typing import Union
from dataclasses import dataclass

import gym
import gym.spaces
from PIL import Image
import numpy as np

from miniwob_interface.action import (
    MiniWoBClick,
    MiniWoBMove,
    MiniWoBRelease,
    MiniWoBType,
    MiniWoBAction,
)
from miniwob_interface.environment import MiniWoBEnvironment

cur_path_dir = os.path.dirname(os.path.realpath(__file__))
miniwob_dir = os.path.join(cur_path_dir, "miniwob_interface", "html", "miniwob")


class MiniWoBEnv(MiniWoBEnvironment, gym.Env):
    """

    ### Observation Space

    The observation is a screen image (width x height x RGB) = (160 x 210 x 3)

    ### Action Space

    There are 3 discrete deterministic actions:

    | Num | Observation
    |-----|-------------------------------------------------------------
    | 0   | Move cursor up
    | 1   | Move cursor down
    | 2   | Move cursor left
    | 3   | Move cursor right
    | 4   | Mouse click and hold
    | 5   | Mouse release
    | 6   | Type a
    | 7   | Type b
    | ... | ...
    | 30  | Type y
    | 31  | Type z


    ### Reward
    1 if success, otherwise 0.
    """

    def __init__(
        self,
        env_name: str,
        seeds: Union[list[int], None] = None,
        num_instances: int = 1,
        miniwob_dir: str = miniwob_dir,
        headless: bool = False,
    ):
        if seeds is None:
            seeds = [1 for _ in range(num_instances)]

        super().__init__(env_name)
        self.base_url = f"file://{miniwob_dir}"
        self.configure(
            num_instances=num_instances,
            seeds=seeds,
            base_url=self.base_url,
            headless=headless,
        )

        @dataclass
        class Loc:
            left: int = 0
            top: int = 0  # Set default value

        # Current location of cursor
        self.cursor_loc: list[Loc] = [Loc(0, 0) for _ in range(num_instances)]

        self.obs_im_width = 160
        self.obs_im_height = 210
        self.num_channels = 3  # RGB
        self.obs_im_size = (self.obs_im_width, self.obs_im_height)

        self.obs_im_shape = self.obs_im_size

        self.observation_space = gym.spaces.Box(
            0,
            255,
            (self.obs_im_width, self.obs_im_height, self.num_channels),
            dtype=int,
        )
        self.action_space = gym.spaces.Discrete(32)

    def reset(
        self,
        seeds: Union[list[int], None] = None,
        mode=None,
        record_screenshots: bool = False,
    ) -> list:
        """Forces stop and start all instances.

        Args:
            seeds (list[object]): Random seeds to set for each instance;
                If specified, len(seeds) must be equal to the number of instances.
                A None entry in the list = do not set a new seed.
            mode (str): If specified, set the data mode to this value before
                starting new episodes.
            record_screenshots (bool): Whether to record screenshots of the states.
        Returns:
            states (list[MiniWoBState])
        """
        seeds = [1 for _ in range(len(self.instances))]
        miniwob_state = super().reset(seeds, mode, record_screenshots)

        return [
            state.screenshot.resize(self.obs_im_shape, Image.ANTIALIAS)
            for state in miniwob_state
        ]

    def step(
        self, actions: list[int]
    ) -> tuple[list[Image.Image], list[float], list[bool], dict]:
        """Applies an action on each instance and returns the results.

        Args:
            actions (int)

        Returns:
            tuple (states, rewards, dones, info)
                states (list[PIL.Image.Image])
                rewards (list[float])
                dones (list[bool])
                info (dict): additional debug information.
                    Global debug information is directly in the root level
                    Local information for instance i is in info['n'][i]
        """

        def convert2webaction(idx: int, action: int) -> Union[MiniWoBAction, None]:
            if action == 0:
                self.cursor_loc[idx].top -= 1
                self.cursor_loc[idx].top = np.clip(
                    self.cursor_loc[idx].top, 0, self.obs_im_height
                )
                return None
            elif action == 1:
                self.cursor_loc[idx].top += 1
                self.cursor_loc[idx].top = np.clip(
                    self.cursor_loc[idx].top, 0, self.obs_im_height
                )
                return None
            elif action == 2:
                self.cursor_loc[idx].left -= 1
                self.cursor_loc[idx].left = np.clip(
                    self.cursor_loc[idx].left, 0, self.obs_im_width
                )
                return None
            elif action == 3:
                self.cursor_loc[idx].left += 1
                self.cursor_loc[idx].left = np.clip(
                    self.cursor_loc[idx].left, 0, self.obs_im_width
                )
                return None
            elif action == 4:
                return MiniWoBClick(self.cursor_loc[idx].left, self.cursor_loc[idx].top)
            elif action == 5:
                return MiniWoBRelease(
                    self.cursor_loc[idx].left, self.cursor_loc[idx].top
                )

            if 6 <= action <= 31:
                return MiniWoBType(chr(ord("a") + action - 6))
            else:
                raise ValueError("Invalid action")

        web_actions = [
            convert2webaction(idx, action) for idx, action in enumerate(actions)
        ]

        states, rewards, dones, info = super().step(web_actions)

        # Obtain screenshot & Resize image obs to match config
        img_states = [
            state.screenshot.resize(self.obs_im_shape) if not dones[i] else None
            for i, state in enumerate(states)
        ]
        return img_states, rewards, dones, info


if __name__ == "__main__":
    env = MiniWoBEnv("click-pie")
    for _ in range(1):
        obs = env.reset(record_screenshots=True)

        # Move cursor to the center of the screen
        for _ in range(90):
            obs, reward, done, info = env.step([3])
        for _ in range(150):
            obs, reward, done, info = env.step([1])

        done = [False]
        while not all(done):
            # Click
            action = [4]
            obs, reward, done, info = env.step(action)
            obs, reward, done, info = env.step([3])
            for ob in obs:
                if ob is not None:
                    ob.show()
            action = [5]
            obs, reward, done, info = env.step(action)

    env.close()
