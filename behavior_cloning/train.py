import argparse

from behavior_cloning import BehaviorCloning

parser = argparse.ArgumentParser(description='.')
parser.add_argument('--epochs', default=30, type=int, help='.')

args = parser.parse_args()

if __name__ == "__main__":
    bc_agent = BehaviorCloning(epochs=args.epochs)
    bc_agent.train()
