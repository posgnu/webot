import gym
import computergym
import random
import numpy as np

NUM_OF_WORKERS = 1
NUM_OF_EPISODES = 10


def solution(state: list) -> tuple[list[int], list[int]]:
    dom_elements = state[0].dom_elements
    field = state[0].fields["target"]

    return [40], [40]


def random_click_action():
    action = [2, random.randint(0, 159), random.randint(0, 159)]
    return action


if __name__ == "__main__":
    env = gym.make(
        "MiniWoBEnv-v0", env_name="click-button", num_instances=NUM_OF_WORKERS
    )

    for _ in range(NUM_OF_EPISODES):
        seeds = [random.random() for _ in range(NUM_OF_WORKERS)]
        states = env.reset(seeds=seeds, record_screenshots=True)

        done = False
        while not done:
            x_coordinates, y_coordinates = solution(states)
            # Click middle point
            actions = [[2, x, y] for x, y in zip(x_coordinates, y_coordinates)]
            assert len(actions) == NUM_OF_WORKERS
            states, reward, dones, info = env.step(actions)
            done = all(dones)

            for state in states:
                if state is not None:
                    state.screenshot.crop((0, 50, 160, 210)).show()
            import time

            time.sleep(3)
    env.close()
