import tensorflow as tf
from src.model_base import ModelBase


class RNN(ModelBase):

    def __init__(self, mode, rnn_type, num_layers, sequence_length, use_bidirectional, dim_hidden, num_labels):
        self.mode = mode
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.use_bidirectional = use_bidirectional
        self.sequence_length = sequence_length
        self.dim_hidden = dim_hidden

        self.num_labels = num_labels

        # self.x = tf.placeholder(tf.float32, shape=[None, self.sequence_length, self.dim_hidden], name='x')
        self.x = tf.placeholder(tf.float32, shape=[None, self.sequence_length, 7], name='x')
        if self.mode == 0:
            self.y = tf.placeholder(tf.int64, shape=[None, self.sequence_length], name='y')
        elif self.mode == 2:
            self.y = tf.placeholder(tf.int64, shape=[None], name='y')

        # self.batch_size = tf.placeholder(tf.int32, shape=[None], name='sequence_length')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        cell_type = {
            'lstm': tf.contrib.rnn.LSTMCell,
            'gru': tf.contrib.rnn.GRUCell
        }

        with tf.variable_scope('dynamic_rnn') as scope:
            stacked_rnn = list()
            for i in range(num_layers):
                # print(i)
                # cell = tf.contrib.rnn.LSTMCell(num_units=dim_hidden, state_is_tuple=True, activation=tf.tanh)
                cell = cell_type[rnn_type](num_units=self.dim_hidden, activation=tf.tanh)
                cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=sequence_length)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0,
                                                     output_keep_prob=self.dropout_keep_prob)
                stacked_rnn.append(cell)

            print('Num of layers: %i' % len(stacked_rnn))
            stacked_rnn = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn)
            # init_state = stacked_rnn.zero_state(batch_size=)

            if self.use_bidirectional:
                stacked_rnn_bw = list()
                for i in range(num_layers):
                    # cell_bi = tf.contrib.rnn.LSTMCell(num_units=dim_hidden, state_is_tuple=True, activation=tf.tanh)
                    cell_bi = cell_type[rnn_type](num_units=self.dim_hidden, activation=tf.tanh)
                    cell_bi = tf.contrib.rnn.AttentionCellWrapper(cell_bi, attn_length=sequence_length)
                    cell_bi = tf.nn.rnn_cell.DropoutWrapper(cell=cell_bi, input_keep_prob=1.0,
                                                            output_keep_prob=self.dropout_keep_prob)
                    stacked_rnn_bw.append(cell_bi)

                stacked_rnn_bw = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw)

                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked_rnn, cell_bw=stacked_rnn_bw,
                                                                         # sequence_length=self.batch_size,
                                                                         # sequence_length=[sequence_length] * self.batch_size,
                                                                         inputs=self.x, dtype=tf.float32, scope=scope)
                outputs = tf.concat(outputs, 2)
                # output_states = tf.concat(output_states, 2)
                self.dim_hidden = self.dim_hidden * 2

            else:
                outputs, output_states = tf.nn.dynamic_rnn(cell=stacked_rnn, inputs=self.x, dtype=tf.float32,
                                                           # sequence_length=self.batch_size,
                                                           # sequence_length=[sequence_length] * self.batch_size,
                                                           scope=scope)

        self.outputs = outputs
        batch_size = tf.shape(outputs)[0]

        with tf.variable_scope('logits') as scope:
            self.logits = self._logits(mode, outputs, batch_size)

        with tf.name_scope('loss'):
            self.loss = self._loss(mode, self.logits, batch_size)

        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope('evaluation'):
            if mode == 0:
                self.pred = tf.argmax(self.logits, axis=2)

                # print(self.y) # Tensor("y:0", shape=(?, 10=sequence_length), dtype=int64, device=/device:CPU:0)
            elif mode == 2:
                self.pred = tf.argmax(self.logits, 1)
                # print(self.pred) # Tensor("evaluation/ArgMax:0", shape=(?,), dtype=int64)
                # print(self.y) # Tensor("y:0", shape=(?, 1), dtype=int64)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.y), tf.float32))

            self.correct_count = tf.reduce_sum(tf.to_float(tf.equal(self.pred, self.y)), axis=0)

    def _logits(self, mode, outputs, batch_size):
        if mode == 0: # Many to many
            X_for_fc = tf.reshape(outputs, [-1, self.dim_hidden])
            outputs2 = tf.contrib.layers.fully_connected(
                inputs=X_for_fc, num_outputs=self.num_labels, activation_fn=None)

            # print(outputs2) # Tensor("logits/fully_connected/BiasAdd:0", shape=(?, 4), dtype=float32, device=/device:CPU:0)
            # reshape out for sequence_loss
            logits = tf.reshape(outputs2, [batch_size, self.sequence_length, self.num_labels])
            # print(logits) Tensor("logits/Reshape_1:0", shape=(?, 10, 4), dtype=float32, device=/device:CPU:0)
        elif mode == 1:
            print('Not implemented yet')
        elif mode == 2: # Many to one
            last_outputs = outputs[:, -1]

            w = self.weight_variable(shape=[self.dim_hidden, self.num_labels])
            b = self.bias_variable(shape=[self.num_labels])

            # self.w = tf.get_variable(dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
            #                           shape=[self.dim_hidden, self.num_labels], name='weight_final')
            # self.b = tf.get_variable(dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
            #                           shape=[self.num_labels], name='bias_final')

            logits = tf.matmul(last_outputs, w) + b

            # X_for_fc = tf.reshape(self.outputs, [-1, dim_hidden])
            # self.logits = tf.contrib.layers.fully_connected(X_for_fc, self.num_labels, activation_fn=None)
        else:
            print('Wrong mode option')

        return logits

    def _loss(self, mode, logits, batch_size):
        # print(self.y)
        # print(self.logits)
        # self.loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits))
        if mode == 0:
            weights = tf.ones([batch_size, self.sequence_length])
            sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=self.y, weights=weights)
            # print(sequence_loss) # Tensor("loss/sequence_loss/truediv:0", shape=(), dtype=float32, device=/device:CPU:0)
            loss = tf.reduce_mean(sequence_loss)
        elif mode == 1:
            print('Not implemented yet')
        elif mode == 2:
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
        else:
            print('Wrong mode option')

        return loss
