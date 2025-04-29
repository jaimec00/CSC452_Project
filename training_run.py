"""
for training the model does epoch loops and 
"""

from data_loader import DataLoader
from box import Box
import yaml
from pathlib import Path


class TrainingRun():
    def __init__(self, config):
        self.data = DataLoader(config.data.train_val_path, config.data.test_path, config.data.annotations_path)
        self.epochs = config.training_params.epochs
        self.lr = config.training_params.lr

        # for 
        for img, label in self.data.test:
            print(img, label)
            break



if __name__ == "__main__":

    config_data: dict = yaml.safe_load(Path("config.yml").read_text())
    config = Box(config_data, default_box=True, conversion_box=True)
    TrainingRun(config)