from abc import abstractmethod


class DataLoader:
    @abstractmethod
    def load_train_data(self):
        pass

    @abstractmethod
    def load_test_data(self):
        pass

    @abstractmethod
    def load_validation_data(self):
        pass

    @abstractmethod
    def load_single_image(self, img_path, target_size):
        pass