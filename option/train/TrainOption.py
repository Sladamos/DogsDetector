import os
from abc import ABC

from initializers.trainer.TensorflowTrainerInitializer import TensorflowTrainerInitializer
from option.Option import Option


class TrainOption(Option, ABC):
    def __init__(self):
        self.initializers = {
            "tensorflow": lambda: TensorflowTrainerInitializer()
        }

    def initialize_trainer(self, app_config):
        initializer_type = app_config["trainer_type"]
        initializer_config = self.create_initializer_config(app_config)
        initializer = self.initializers[initializer_type]
        return initializer.initialize_trainer(initializer_config)

    def calculate_number_of_classes(self, app_config):
        path = os.path.normpath(app_config["training_set_path"])
        dirs = os.listdir(path)
        return len(dirs)

    def create_initializer_config(self, app_config):
        number_of_classes = self.calculate_number_of_classes(app_config)
        initializer_config = {
            "train_from_existing": app_config["train_from_existing"],
            "number_of_classes": number_of_classes,
            "detected_type": app_config["detected_type"]
        }
        return initializer_config
