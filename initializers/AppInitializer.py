from abc import abstractmethod


class AppInitializer:

    @abstractmethod
    def initialize_app(self, detected_type):
        pass