"""
for training the model does epoch loops and 
"""

from data_loader import DataLoader
from box import Box
import yaml
from pathlib import Path
from model import setup_model

class TrainingRun():
    def __init__(self, config):
        self.data = DataLoader(config.data.train_val_path, config.data.test_path, config.data.annotations_path, config.data.num_train, config.data.num_val, config.data.num_test)
        self.training_params = config.training_params
        self.model = setup_model(config.hyper_params)
        self.model.compile(loss=self.training_params.loss_func,
                            optimizer = SGD(self.training_params.lr),
                            metrics = ['accuracy']
                            )

    def train(self):
        history = model.fit(self.data.train.imgs, self.data.train.labels,
                    batch_size = self.training_params.batch_size,
                    epochs = self.training_params.epochs,
                    verbose = 1,
                    validation_data = (self.data.val.imgs, self.data.val.labels)
                    )

if __name__ == "__main__":

    config_data: dict = yaml.safe_load(Path("config.yml").read_text())
    config = Box(config_data, default_box=True, conversion_box=True)
    training_run = TrainingRun(config)
    training_run.train()
    training_run.test()