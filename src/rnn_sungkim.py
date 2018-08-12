import tensorflow as tf
from .model_base import ModelBase


class RNN(ModelBase):
    def __init__(self, num_layers, sequence_length, use_bidirectional, dim_hidden, num_labels):
        output_dim = 1

        X = tf.placeholder(tf.float32, [None, sequence_length, 7])
        Y = tf.placeholder(tf.float32, [None, 1])

        # build a LSTM network
        cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=dim_hidden, state_is_tuple=True, activation=tf.tanh)
        outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        Y_pred = tf.contrib.layers.fully_connected(
            outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

        # cost/loss
        loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(loss)

        # RMSE
        targets = tf.placeholder(tf.float32, [None, 1])
        predictions = tf.placeholder(tf.float32, [None, 1])
        rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))