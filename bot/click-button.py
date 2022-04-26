import gym
import computergym
import random

from utils import NUM_OF_WORKERS, NUM_OF_EPISODES, Recorder

def solution(states: list) -> tuple[list[int], list[int]]:
    x_coordinates, y_coordinates = [], []
    for state in states:
        dom_elements = state.dom_elements
        field = state.fields["target"]

        for element in dom_elements:
            if field == element.text:
                left, top, width, height = (
                    element.left,
                    element.top,
                    element.width,
                    element.height,
                )
                break
        else:
            raise ValueError("Cannot find solution")

        x_coordinates.append(left + width / 2)
        y_coordinates.append(top + height / 2)

    assert len(states) == len(x_coordinates) == len(y_coordinates)

    return x_coordinates, y_coordinates

if __name__ == "__main__":
    env = gym.make(
        "MiniWoBEnv-v0", env_name="click-button", num_instances=NUM_OF_WORKERS
    )

    recorder = Recorder("click-button")

    for _ in range(NUM_OF_EPISODES):
        seeds = [random.random() for _ in range(NUM_OF_WORKERS)]
        states = env.reset(seeds=seeds, record_screenshots=True)

        done = False
        while not done:
            x_coordinates, y_coordinates = solution(states)
            actions = [
                [computergym.utils.action_type_to_number("click"), x, y]
                for x, y in zip(x_coordinates, y_coordinates)
            ]
            assert len(actions) == NUM_OF_WORKERS

            recorder.append_data(states, actions)

            states, reward, dones, info = env.step(actions)
            done = all(dones)

            import time

            time.sleep(1)
    recorder.save()
    env.close()
