import gym
import computergym

if __name__ == "__main__":
    env = gym.make("MiniWoBEnv-v0", env_name="click-pie")

    for _ in range(1):
        states = env.reset(record_screenshots=True)

        done = [False]
        while not all(done):
            # Click middle point
            actions = [[0, 80, 140], [1, 80, 140]]
            for action in actions:
                states, reward, done, info = env.step([action])

            for state in states:
                if state is not None:
                    # state.screenshot.show()
                    pass
            import code

            code.interact(local=locals())
            import time

            time.sleep(1)
    env.close()
