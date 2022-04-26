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
    find_element_by_text,
)


def solution(current_state: int, states: list) -> tuple[list[int], list[int]]:
    x_coordinates, y_coordinates = [], []
    for state in states:
        dom_elements = state.dom_elements
        elements = find_element_by_tag("button", dom_elements)

        if current_state == 0:
            # Handle the case when button two is overlapped by button one
            one_button = find_element_by_text("ONE", elements)[0]
            one_x, one_y = get_click_point(one_button)

            two_button = find_element_by_text("TWO", elements)[0]
            two_x, two_y = get_click_point(two_button)

            delta_x = two_x - one_x
            delta_y = two_y - one_y

            if abs(delta_x) < 21 and abs(delta_y) < 21:
                if delta_x < 0:
                    one_x = one_x + 20 + (delta_x / 2)
                else:
                    one_x = one_x - 20 + (delta_x / 2)

                if delta_y < 0:
                    one_y = one_y + 20 + (delta_y / 2)
                else:
                    one_y = one_y - 20 + (delta_y / 2)

            x_coordinates.append(one_x)
            y_coordinates.append(one_y)
        elif current_state == 1:
            two_button = find_element_by_text("TWO", elements)[0]

            x, y = get_click_point(two_button)
            x_coordinates.append(x)
            y_coordinates.append(y)
        else:
            raise ValueError("cannot find solution")

    assert len(states) == len(x_coordinates) == len(y_coordinates)

    return x_coordinates, y_coordinates


if __name__ == "__main__":
    env_name = "click-button-sequence"
    env = gym.make("MiniWoBEnv-v0", env_name=env_name, num_instances=NUM_OF_WORKERS)

    recorder = Recorder(env_name)

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

            time.sleep(0.5)
            current_state += 1
    recorder.save()
    env.close()
