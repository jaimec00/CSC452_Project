"""
for training the model 
"""

from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from data_loader import DataLoader
from model import setup_model
from pathlib import Path
from box import Box
import yaml

class TrainingRun():

    def __init__(self, config):

        self.training_params = config.training_params
        self.model = setup_model(   input_shape=(   config.hyper_params.input_size.height, 
                                                    config.hyper_params.input_size.width, 
                                                    config.hyper_params.input_size.channels
                                                ), 
                                    num_layers=config.hyper_params.unet.num_layers, 
                                    initial_filters=config.hyper_params.unet.initial_filters
                                )
        self.model.compile( loss=self.training_params.loss_func,
                            optimizer = Adam(learning_rate=self.training_params.lr),
                            metrics = ['accuracy']
                        )

        self.model.summary()

        self.data = DataLoader( config.data.train_val_path, config.data.test_path, 
                                config.data.annotations_path, config.data.num_train, 
                                config.data.num_val, config.data.num_test
                            )

        self.out_path = Path(config.output.out_path)
        self.out_path.mkdir(exist_ok=True, parents=True)


    def train(self):
        self.history = self.model.fit(  self.data.train.imgs, self.data.train.labels,
                                        batch_size = self.training_params.batch_size,
                                        epochs = self.training_params.epochs,
                                        verbose = 1,
                                        validation_data = (self.data.val.imgs, self.data.val.labels)
                                    )

        self.plot_training()
        self.model.save(self.out_path / Path("model.keras"))

    def plot_training(self):

        # Loss
        plt.figure()
        plt.plot(self.history.history['loss'],    label='train loss')
        plt.plot(self.history.history['val_loss'],label='val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.out_path / Path("loss_v_epoch.png"))

        # Accuracy
        plt.figure()
        plt.plot(self.history.history['accuracy'],    label='train acc')
        plt.plot(self.history.history['val_accuracy'],label='val acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(self.out_path / Path("acc_v_epoch.png"))

    def test(self):
        test_loss, test_acc = self.model.evaluate(
                                self.data.test.imgs,
                                self.data.test.labels,
                                batch_size=self.training_params.batch_size,
                                verbose=1
                            )
        print(f"Test loss:  {test_loss:.4f}")
        print(f"Test acc:   {test_acc:.4f}")

if __name__ == "__main__":

    config_data: dict = yaml.safe_load(Path("config.yml").read_text())
    config = Box(config_data, default_box=True, conversion_box=True)

    training_run = TrainingRun(config)
    training_run.train()
    training_run.test()