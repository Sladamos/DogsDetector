from trainers.Trainer import Trainer


class SimpleTrainer(Trainer):

    def create_raw_model(self, models_creator, input_shape):
        return models_creator.create_simple_neural_model(input_shape=input_shape)
