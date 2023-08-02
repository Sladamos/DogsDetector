import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, plotter_config):
        self.results = None
        self.plotter_config = plotter_config

    def print_plots(self, results):
        accuracy_config = self.plotter_config["accuracy"]
        loss_config = self.plotter_config["loss"]
        self.results = results
        if accuracy_config["draw_plot"]:
            self.__print_accuracy_plot(accuracy_config)
        if loss_config["draw_plot"]:
            self.__print_loss_plot(loss_config)
        self.results = None

    def __print_accuracy_plot(self, accuracy_config):
        plt.plot(self.results.history['accuracy'])
        plt.plot(self.results.history['val_accuracy'])
        plt.title('Accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['training', 'validation'], loc='upper left')
        if accuracy_config["to_file"]:
            file_path = accuracy_config["file_path"]
            plt.savefig(file_path)
        plt.show()

    def __print_loss_plot(self, loss_config):
        plt.plot(self.results.history['loss'])
        plt.plot(self.results.history['val_loss'])
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['training', 'validation'], loc='upper left')
        if loss_config["to_file"]:
            file_path = loss_config["file_path"]
            plt.savefig(file_path)
        plt.show()
