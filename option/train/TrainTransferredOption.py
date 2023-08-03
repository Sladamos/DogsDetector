from option.train.TrainOption import TrainOption
from trainers.TransferredTrainer import TransferredTrainer


class TrainTransferredOption(TrainOption):

    def execute(self, config):
        app_config = config["transferred"]
        trainer = TransferredTrainer()
        self.execute_training(app_config, trainer)
