from abc import abstractmethod


class DataLoader:
    @abstractmethod
    def load_train_images(self):
        pass

    @abstractmethod
    def load_test_images(self):
        pass

    @abstractmethod
    def load_image(self):
        pass