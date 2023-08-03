from callbacks.CallbacksCreator import CallbacksCreator
import tensorflow as tf


class TensorflowCallbacksCreator(CallbacksCreator):
    def create_checkpoint(self, checkpoint_config):
        return tf.keras.callbacks.ModelCheckpoint(checkpoint_config["checkpoint_path"], monitor='val_accuracy',
                                                  save_best_only=True)
