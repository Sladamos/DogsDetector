from abc import abstractmethod


class Option:

    @abstractmethod
    def execute(self, config):
        pass
