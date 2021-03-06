import argparse

from behavior_cloning import BehaviorCloning

parser = argparse.ArgumentParser(description=".")
parser.add_argument("--epochs", default=200, type=int, help=".")

args = parser.parse_args()

if __name__ == "__main__":
    bc_agent = BehaviorCloning(env_name="click-button", epochs=args.epochs)

    train_loader, val_order = bc_agent.create_datasets()

    bc_agent.build_vocab()

    # bc_agent.restore()
    bc_agent.build_model()

    bc_agent.train_model(train_loader, val_order)
