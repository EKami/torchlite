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


class SimpleCnn(BaseModel):
    def __init__(self, logger, num_classes):
        """

        Args:
            logger (logger): A Python logger
            num_classes (int): Number of output class
        """
        super(BaseModel, self).__init__(logger, 'simple_cnn', num_classes)
        self.conv_1 = keras.layers.Conv2D(filters=32, kernel_size=(5, 5),
                                          padding='Same', activation='relu')
        self.bn_1 = keras.layers.BatchNormalization()
        self.conv_2 = keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                                          activation='relu')
        self.bn_2 = keras.layers.BatchNormalization()
        self.max_pool_1 = keras.layers.MaxPool2D(pool_size=(2, 2))

        self.conv_3 = keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.bn_3 = keras.layers.BatchNormalization()
        self.conv_4 = keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.bn_4 = keras.layers.BatchNormalization()
        self.conv_5 = keras.layers.Conv2D(128, (3, 3), activation='relu')
        self.bn_5 = keras.layers.BatchNormalization()
        self.conv_6 = keras.layers.Conv2D(256, (3, 3), activation='relu')
        self.bn_6 = keras.layers.BatchNormalization()
        self.max_pool_2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.flatten1 = keras.layers.Flatten()
        self.dense_1 = keras.layers.Dense(256, activation='relu')
        self.drop_1 = keras.layers.Dropout(0.5)
        self.dense_2 = keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.max_pool_1(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.conv_6(x)
        x = self.bn_6(x)
        x = self.max_pool_2(x)

        x = self.flatten1(x)
        x = self.dense_1(x)
        x = self.drop_1(x)
        x = self.dense_2(x)
        # Logits are actually sigmoid output as output without sigmoid is useless
        return x
