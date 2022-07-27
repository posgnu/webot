import gym
import computergym
import random

from utils import NUM_OF_WORKERS, NUM_OF_EPISODES, ODRecorder, COMPONENT_TO_CLASS


def extract_boxes(states: list) -> tuple[list, list]:
    image_list = []
    label_list = []

    for state in states:
        image_list.append(state.screenshot)

        dom_elements = state.dom_elements
        label = []
        for element in dom_elements:
            if element.tag in COMPONENT_TO_CLASS.keys():
                cls = COMPONENT_TO_CLASS[element.tag]

                left, top, width, height = (
                    element.left - 1,
                    element.top - 1,
                    element.width + 2,
                    element.height + 2,
                )
                center_x = left + width / 2
                center_y = top + height / 2

                label.append(
                    [cls, center_x / 160, center_y / 210, width / 160, height / 210]
                )

        label_list.append(label)

    assert len(states) == len(label_list)

    return image_list, label_list


if __name__ == "__main__":
    env = gym.make(
        "MiniWoBEnv-v0", env_name="click-button", num_instances=NUM_OF_WORKERS
    )

    recorder = ODRecorder("click-button")

    for _ in range(NUM_OF_EPISODES):
        seeds = [random.random() for _ in range(NUM_OF_WORKERS)]
        states = env.reset(seeds=seeds, record_screenshots=True)

        images, labels = extract_boxes(states)

        assert len(images) == NUM_OF_WORKERS

        recorder.append_observation(images, labels)

        import time

        time.sleep(1)
    env.close()
