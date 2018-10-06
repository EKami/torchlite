import tensorflow as tf
from tensorflow import keras


class BaseModel(keras.Model):
    def __init__(self, logger, name, num_classes):
        super(BaseModel, self).__init__(name=name)
        self.logger = logger
        self.num_classes = num_classes

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
