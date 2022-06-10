import os
import gzip
import pickle
import numpy as np
import computergym

NUM_OF_WORKERS = 12
NUM_OF_EPISODES = 5000

DATA_FILE = "bot_data.gzip"
DATA_DIR = "../data"


def find_element_by_tag(tag: str, dom_elements: list):
    elements = []
    for element in dom_elements:
        if element.tag == tag:
            elements.append(element)

    return elements


def find_element_by_text(text: str, dom_elements: list):
    elements = []
    for element in dom_elements:
        if element.text == text:
            elements.append(element)

    return elements


def random_click_action():
    action = [
        computergym.utils.action_type_to_number("click"),
        random.randint(0, 159),
        random.randint(0, 159),
    ]
    return action


def get_click_point(element) -> tuple[int, int]:
    left, top, width, height = (
        element.left,
        element.top,
        element.width,
        element.height,
    )
    return left + width / 2, top + height / 2


class Recorder:
    def __init__(self, env_name: str) -> None:
        self.datafile_path = os.path.join(DATA_DIR, env_name + "_" + DATA_FILE)
        try:
            with gzip.open(self.datafile_path, "rb") as f:
                self.observations = pickle.load(f)
        except IOError:
            self.observations = []

    def append_data(self, states: list, actions: list):
        for state, action in zip(states, actions):
            rgb_list = np.array(state.screenshot).reshape((-1, 3))
            # 160 x 210 = 33600
            assert rgb_list.shape == (33600, 3)
            utterance = state.utterance

            state_dict_repr = {"utterance": utterance, "img": rgb_list}

            action_dict_repr = {
                "type": computergym.utils.action_number_to_type(action[0]),
                "x": action[1],
                "y": action[2],
            }

            self.observations.append((state_dict_repr, action_dict_repr))

    def save(self):
        with gzip.open(self.datafile_path, "wb") as f:
            pickle.dump(self.observations, f)
