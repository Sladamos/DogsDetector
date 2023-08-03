from abc import abstractmethod


class CallbacksCreator:

    @abstractmethod
    def create_checkpoint(self, checkpoint_config):
        pass
