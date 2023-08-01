from abc import abstractmethod


class TrainerInitializer:
    @abstractmethod
    def initialize_trainer(self, initializer_config):
        pass
