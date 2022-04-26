from locale import locale_alias
import gym
import computergym
import random

from utils import (
    NUM_OF_WORKERS,
    NUM_OF_EPISODES,
    Recorder,
    get_click_point,
    find_element_by_tag,
)



def solution(current_state: int, states: list) -> tuple[list[int], list[int]]:
    x_coordinates, y_coordinates = [], []
    for state in states:
        dom_elements = state.dom_elements
        element = find_element_by_tag("select", dom_elements)[0]

        if current_state == 0:
            x, y = get_click_point(element)
            x_coordinates.append(x)
            y_coordinates.append(y)
        elif current_state == 1:
            field = state.fields["target"]
            options = element.classes.split(",")
            option_idx = options.index(field)
            import code

            code.interact(local=locals())

            x, y = get_click_point(element)
            x_coordinates.append(x)
            y_coordinates.append(y + 22 * option_idx)
        elif current_state == 2:
            btn = find_tag_element("button", dom_elements)
            x, y = get_click_point(btn)
            x_coordinates.append(x)
            y_coordinates.append(y)
        else:
            raise ValueError("cannot find solution")

    assert len(states) == len(x_coordinates) == len(y_coordinates)

    return x_coordinates, y_coordinates


if __name__ == "__main__":
    env = gym.make(
        "MiniWoBEnv-v0", env_name="choose-list", num_instances=NUM_OF_WORKERS
    )

    recorder = Recorder("choose-list")

    for _ in range(NUM_OF_EPISODES):
        seeds = [random.random() for _ in range(NUM_OF_WORKERS)]
        states = env.reset(seeds=seeds, record_screenshots=True)

        done = False
        current_state = 0
        while not done:
            x_coordinates, y_coordinates = solution(current_state, states)
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
            current_state += 1
    recorder.save()
    env.close()
