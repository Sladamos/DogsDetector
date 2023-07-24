from abc import abstractmethod


class DetectorsFactory:
    @abstractmethod
    def create_detector(self):
        pass