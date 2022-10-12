import pytorchConfig
from onespace.pytorch.cv import Experiment

def training(config):
    exp = Experiment(config)
    exp.run_experiment()


if __name__ == "__main__":
    training(pytorchConfig)
