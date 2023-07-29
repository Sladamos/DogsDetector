from abc import ABC

from initializers.TensorflowAppInitializer import TensorflowAppInitializer
from option.Option import Option


class AppOption(Option, ABC):
    def __init__(self):
        self.initializers = {
            "tensorflow": TensorflowAppInitializer
        }

    def initialize_app(self, init_config, app_config):
        initializer_type = app_config["initializer_type"]
        detected_type = app_config["detected_type"]
        initializer = self.initializers[initializer_type](init_config)
        return initializer.initialize_app(detected_type)
