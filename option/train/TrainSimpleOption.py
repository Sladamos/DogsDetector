from option.train.TrainOption import TrainOption


class TrainSimpleOption(TrainOption):
    def execute(self, config):
        batch_size = 64
        path = os.path.normpath("./images/all_dogs")
        dirs = os.listdir(path)
        models_creator = DogsModelsCreator(len(dirs))
        data_loader = DogsDataLoader(batch_size, path)
        cnn_model = models_creator.create_simple_neural_model(input_shape=(224, 224, 3))
        # loader = TensorModelLoader()
        # cnn_model = loader.load_model("./newHope/saved/our")
        saver = TensorModelSaver()
        train_data = data_loader.load_train_data()
        validation_data = data_loader.load_validation_data()
        epochs = 150
        verbose = 1

        results = cnn_model.train_with_validation(train_data, validation_data, epochs=epochs, batch_size=batch_size,
                                                  verbose=verbose, callbacks=[])
        make_plots(results, "simple.png")