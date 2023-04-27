from abc import abstractmethod


class DataLoader:
    @abstractmethod
    def load_train_data(self):
        pass

    @abstractmethod
    def load_test_data(self):
        pass

    @abstractmethod
    def load_single_data(self):
        pass