import tensorflow as tf
from .model_base import ModelBase


class RNN(ModelBase):

    def __init__(self, num_layers, sequence_length, use_bidirectional, dim_hidden, num_labels):
        self.num_layers = num_layers
        self.use_bidirectional = use_bidirectional
        self.sequence_length = sequence_length
        self.dim_hidden = dim_hidden
        if use_bidirectional:
            self.dim_hidden = self.dim_hidden * 2
        self.num_labels = num_labels
        output_dim = 1

        # self.x = tf.placeholder(tf.float32, shape=[None, self.sequence_length, self.dim_hidden], name='x')
        self.x = tf.placeholder(tf.float32, shape=[None, self.sequence_length, 7], name='x')
        self.y = tf.placeholder(tf.int64, shape=[None], name='y')

        # self.batch_size = tf.placeholder(tf.int32, shape=[None], name='sequence_length')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        with tf.variable_scope('dynamic_rnn') as scope:
            stacked_rnn = list()
            for i in range(num_layers):
                # print(i)
                cell = tf.contrib.rnn.LSTMCell(num_units=dim_hidden, state_is_tuple=True, activation=tf.tanh)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0,
                                                     output_keep_prob=self.dropout_keep_prob)
                stacked_rnn.append(cell)

            print('Num of layers: %i' % len(stacked_rnn))
            stacked_rnn = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn)

            if self.use_bidirectional:
                stacked_rnn_bw = list()
                for i in range(num_layers):
                    cell_bi = tf.contrib.rnn.LSTMCell(num_units=dim_hidden, state_is_tuple=True, activation=tf.tanh)
                    cell_bi = tf.nn.rnn_cell.DropoutWrapper(cell=cell_bi, input_keep_prob=1.0,
                                                            output_keep_prob=self.dropout_keep_prob)
                    stacked_rnn_bw.append(cell_bi)

                stacked_rnn_bw = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw)

                self.outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked_rnn, cell_bw=stacked_rnn_bw,
                                                                         # sequence_length=self.batch_size,
                                                                         # sequence_length=[sequence_length] * self.batch_size,
                                                                         inputs=self.x, dtype=tf.float32, scope=scope)
                self.outputs = tf.concat(self.outputs, 2)
                # output_states = tf.concat(output_states, 2)

            else:
                self.outputs, output_states = tf.nn.dynamic_rnn(cell=stacked_rnn, inputs=self.x, dtype=tf.float32,
                                                           # sequence_length=self.batch_size,
                                                           # sequence_length=[sequence_length] * self.batch_size,
                                                           scope=scope)

            # fcn = outputs[self.sequence_length - 1]
            # print(fcn)
            # print(outputs.eval())
            # self.outputs = outputs

            self.last_output = self.outputs[:, -1]

        with tf.variable_scope('logits') as scope:
            self.w = self.weight_variable(shape=[self.dim_hidden, self.num_labels])
            self.b = self.bias_variable(shape=[self.num_labels])

            # self.w = tf.get_variable(dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
            #                           shape=[self.dim_hidden, self.num_labels], name='weight_final')
            # self.b = tf.get_variable(dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
            #                           shape=[self.num_labels], name='bias_final')

            self.logits = tf.matmul(self.last_output, self.w) + self.b

            # X_for_fc = tf.reshape(self.outputs, [-1, dim_hidden])
            # self.logits = tf.contrib.layers.fully_connected(X_for_fc, self.num_labels, activation_fn=None)

        with tf.name_scope('loss'):
            # print(self.y)
            # print(self.logits)
            # self.loss = tf.reduce_mean(
            #     tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits))

            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))

        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope('evaluation'):
            self.pred = tf.argmax(self.logits, 1)
            # print(self.pred) # Tensor("evaluation/ArgMax:0", shape=(?,), dtype=int64)
            # print(self.y) # Tensor("y:0", shape=(?, 1), dtype=int64)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.y), tf.float32))

            self.correct_count = tf.reduce_sum(tf.to_float(tf.equal(self.pred, self.y)), axis=0)





