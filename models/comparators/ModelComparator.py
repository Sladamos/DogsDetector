from data.Data import Data
from models.NeuralModel import NeuralModel
from models.TensorNeuralModel import TensorNeuralModel


class ModelComparator:
    def __init__(self, test_data):
        self.test_data = test_data

    def evaluate_compare(self, model_a, model_b, number_of_samples):

        loss_a, accuracy_a = model_a.evaluate(self.test_data, number_of_samples)
        loss_b, accuracy_b = model_b.evaluate(self.test_data, number_of_samples)
        if accuracy_a > accuracy_b:
            return 1, accuracy_a
        elif accuracy_a == accuracy_b:
            return 0, accuracy_b
        else:
            return -1, accuracy_b

    def predict_compare(self):
        # TODO: implement predict
        pass
