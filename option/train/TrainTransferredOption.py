from option.train.TrainOption import TrainOption


class TrainTransferredOption(TrainOption):
    def get_config_name(self):
        return "transferred"

    def create_raw_model(self, models_creator, input_shape):
        return models_creator.create_advanced_neural_model(input_shape=input_shape)
