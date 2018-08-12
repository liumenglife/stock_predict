import tensorflow as tf


class ModelBase:
    def __init__(self):
        pass

    def weight_variable(self, shape):
        initial = tf.contrib.layers.xavier_initializer()
        return tf.Variable(initial(shape))

    def bias_variable(self, shape):
        initial = tf.contrib.layers.xavier_initializer()
        return tf.Variable(initial(shape))