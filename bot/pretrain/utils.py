import os
import gzip
import pickle
import numpy as np
import computergym
import random

NUM_OF_WORKERS = 1
NUM_OF_EPISODES = 1000

DATA_FILE = "bot_data.gzip"
DATA_DIR = "../data"
PRETRAIN_DATA_DIR = "../../data/pretrain"

COMPONENT_TO_CLASS = {"button": 0}


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


class ODRecorder:
    def __init__(self, env_name: str, start_id=0) -> None:
        self.image_dir = os.path.join(PRETRAIN_DATA_DIR + "/" + env_name, "images")
        self.label_dir = os.path.join(PRETRAIN_DATA_DIR + "/" + env_name, "labels")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        self.id = start_id

    def generate_id(self) -> int:
        new_id = self.id
        self.id += 1
        return new_id

    def append_observation(self, PIL_images: list, labels: list) -> None:
        assert len(PIL_images) == len(labels)

        img_id = self.generate_id()

        for image, label in zip(PIL_images, labels):
            image_path = os.path.join(self.image_dir, str(img_id) + ".jpg")
            image.crop((0, 0, 320, 420)).resize((160, 210)).save(image_path)

            label_path = os.path.join(self.label_dir, str(img_id) + ".txt")
            with open(label_path, "w") as file:
                for box in label:
                    file.write(" ".join(str(x) for x in box))
                    file.write("\n")
