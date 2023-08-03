import os
from abc import ABC, abstractmethod

from displayer.Plotter import Plotter
from initializers.trainer.TensorflowTrainerInitializer import TensorflowTrainerInitializer
from option.Option import Option


class TrainOption(Option, ABC):
    def __init__(self):
        self.initializers = {
            "tensorflow": TensorflowTrainerInitializer
        }

    def execute_training(self, app_config, trainer):
        data_loader, data_normalizer, model_operator, model_saver, callbacks = self.initialize_trainer(app_config)
        model, classes, results = trainer.train_with_config(app_config, model_operator, data_loader, callbacks)
        model_saver.save_model_with_classes(model, classes, app_config["save"])
        self.make_plots(results, app_config["plots"])

    def initialize_trainer(self, app_config):
        initializer_type = app_config["trainer_type"]
        initializer_config = self.create_initializer_config(app_config)
        initializer = self.initializers[initializer_type]()
        return initializer.initialize_trainer(initializer_config)

    def calculate_number_of_classes(self, app_config):
        path = os.path.normpath(app_config["training_set_path"])
        dirs = os.listdir(path)
        return len(dirs)

    def create_initializer_config(self, app_config):
        number_of_classes = self.calculate_number_of_classes(app_config)
        initializer_config = {
            "train_from_existing": app_config["train_from_existing_model"],
            "number_of_classes": number_of_classes,
            "detected_type": app_config["detected_type"],
            "callbacks": app_config["callbacks"]
        }
        return initializer_config

    def make_plots(self, results, plotter_config):
        plotter = Plotter(plotter_config)
        plotter.print_plots(results)

