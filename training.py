import tabularConfig
from src.onespace.tabular.classification.model import Experiment


def training(config):
    exp = Experiment(config)
    exp.run_experiment()


if __name__ == "__main__":
    training(tabularConfig)
