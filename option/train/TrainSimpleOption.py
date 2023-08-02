from option.train.TrainOption import TrainOption


class TrainSimpleOption(TrainOption):
    def get_config_name(self):
        return "simple"

    def create_raw_model(self, models_creator, input_shape):
        return models_creator.create_simple_neural_model(input_shape=input_shape)
