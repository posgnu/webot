import os
from typing import Union
from xml.dom import minicompat

import gym
import gym.spaces
from PIL import Image
import numpy as np

from miniwob.miniwob_interface.action import (
    MiniWoBPress,
    MiniWoBRelease,
    MiniWoBAction,
    MiniWoBCoordClick,
)
from miniwob.miniwob_interface.environment import MiniWoBEnvironment

cur_path_dir = os.path.dirname(os.path.realpath(__file__))
miniwob_dir = os.path.join(cur_path_dir, "miniwob_interface", "html", "miniwob")


class MiniWoBEnv(MiniWoBEnvironment, gym.Env):
    """

    ### Observation Space

    The observation is a screen image (width x height x RGB) = (160 x 210 x 3)

    ### Action Space (action type, x coordinate, y coordinate)

    | type  | action type
    |-------|-------------------------------------------------------------
    | 0     | Mouse click and hold
    | 1     | Mouse release
    | 2     | Mouse click

    |   type  | x coordinate
    |---------|-------------------------------------------------------------
    | 0 ~ 159 | Mouse click and hold


    |   type   | y coordinate
    |----------|-------------------------------------------------------------
    | 0 ~  159 | Mouse click and hold


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
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, 0]), high=np.array([2, 159, 159]), shape=(3,), dtype=int
        )

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
        # seeds = [1 for _ in range(len(self.instances))]
        miniwob_state = super().reset(seeds, mode, record_screenshots)

        for state in miniwob_state:
            if state:
                state.set_screenshot(
                    state.screenshot.resize(self.obs_im_shape, Image.ANTIALIAS)
                )

        return miniwob_state

    def convert2webaction(self, action: list[int]) -> Union[MiniWoBAction, None]:
        assert self.action_space.shape
        assert len(action) == self.action_space.shape[0]

        action_type, x_coordinate, y_coordinate = action
        x_coordinate = np.clip(x_coordinate, 0, 159)
        y_coordinate = np.clip(y_coordinate, 0, 219)
        if action_type == 0:
            return MiniWoBPress(x_coordinate, y_coordinate)
        elif action_type == 1:
            return MiniWoBRelease(x_coordinate, y_coordinate)
        elif action_type == 2:
            return MiniWoBCoordClick(x_coordinate, y_coordinate)
        else:
            raise NotImplemented

    def step(
        self,
        actions: list[list[int]],
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
        assert np.array(actions).shape[1:] == self.action_space.shape

        web_actions = [self.convert2webaction(action) for action in actions]

        states, rewards, dones, info = super().step(web_actions)

        # Obtain screenshot & Resize image obs to match config
        # assert not None in states
        for i, state in enumerate(states):
            if state:
                state.set_screenshot(
                    state.screenshot.resize(self.obs_im_shape) if not dones[i] else None
                )

        return states, rewards, dones, info


if __name__ == "__main__":
    env = MiniWoBEnv("click-pie")
    for _ in range(1):
        obs = env.reset(record_screenshots=True)

        done = [False]
        while not all(done):
            # Click middle point
            actions = [[0, 80, 140], [1, 80, 140]]
            for action in actions:
                obs, reward, done, info = env.step([action])

            for ob in obs:
                if ob is not None:
                    ob.show()
            import time

            time.sleep(3)
    env.close()
