import tensorflow as tf
from .model_base import ModelBase


class RNN(ModelBase):

    def __init__(self, mode, rnn_type, feature_length, num_layers, sequence_length, use_bidirectional,
                 dim_hidden, num_labels, attention_type):
        self.mode = mode
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.use_bidirectional = use_bidirectional
        self.sequence_length = sequence_length
        if self.use_bidirectional:
            self.dim_hidden = dim_hidden * 2

        self.num_labels = num_labels
        self.attention_type = attention_type

        # self.x = tf.placeholder(tf.float32, shape=[None, self.sequence_length, self.dim_hidden], name='x')
        self.x = tf.placeholder(tf.float32, shape=[None, self.sequence_length, feature_length], name='x')
        if self.mode == 0:
            self.y = tf.placeholder(tf.int64, shape=[None, self.sequence_length], name='y')
        elif self.mode == 2:
            self.y = tf.placeholder(tf.int64, shape=[None], name='y')

        # self.batch_size = tf.placeholder(tf.int32, shape=[None], name='sequence_length')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        print('Num of layers: %i' % num_layers)
        batch_size = tf.shape(self.x)[0]

        outputs, output_states = self.build_rnn_layers(use_bidirectional=self.use_bidirectional,
                                                                 num_layers=self.num_layers,
                                                                 rnn_type=self.rnn_type,
                                                                 dim_hidden=dim_hidden,
                                                                 keep_prob=self.dropout_keep_prob)
                                                                 # , batch_size=batch_size)

        self.outputs, self.output_states = self.build_attention(attention_type, outputs, output_states, rnn_type, num_layers,
                                                                self.dim_hidden, self.dropout_keep_prob,
                                                                sequence_length, self.y)

        # batch_size = tf.shape(self.outputs)[0]

        self.logits = self.build_logits(mode, self.outputs, batch_size)
        self.loss = self.build_loss(mode, self.logits, batch_size)
        self.train = self.optimize(learning_rate=self.learning_rate, loss=self.loss)
        self.pred, self.accuracy, self.correct_count = self.evaluate(self.mode, self.logits, self.y)

    def build_rnn_cell_stack(self, rnn_type, num_layers, dim_hidden, keep_prob):
        cell_type = {
            'lstm': tf.contrib.rnn.LSTMCell,
            'gru': tf.contrib.rnn.GRUCell
        }

        cell = lambda x: tf.nn.rnn_cell.DropoutWrapper(cell=x(dim_hidden, activation=tf.tanh),
                                                       input_keep_prob=1.0,
                                                       output_keep_prob=keep_prob)

        stacked_rnn = tf.nn.rnn_cell.MultiRNNCell(cells=[cell(cell_type[rnn_type]) for i in range(num_layers)])
        return stacked_rnn

    def build_rnn_layers(self, use_bidirectional, num_layers, rnn_type, dim_hidden, keep_prob, batch_size=0):

        with tf.variable_scope('dynamic_rnn') as scope:
            stacked_rnn = self.build_rnn_cell_stack(rnn_type=rnn_type, num_layers=num_layers, dim_hidden=dim_hidden,
                                                    keep_prob=keep_prob)
            # init_state = stacked_rnn.zero_state(batch_size=batch_size, dtype=tf.float32)

            if use_bidirectional:
                stacked_rnn_bw = self.build_rnn_cell_stack(rnn_type=rnn_type, num_layers=num_layers,
                                                           dim_hidden=dim_hidden, keep_prob=keep_prob)

                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked_rnn, cell_bw=stacked_rnn_bw,
                                                                         # sequence_length=self.batch_size,
                                                                         # sequence_length=[sequence_length] * self.batch_size,
                                                                         inputs=self.x, dtype=tf.float32, scope=scope)
                outputs = tf.concat(outputs, 2)
                output_states = tf.concat(output_states, 2)

            else:
                outputs, output_states = tf.nn.dynamic_rnn(cell=stacked_rnn, inputs=self.x, dtype=tf.float32,
                                                           # sequence_length=self.batch_size,
                                                           # sequence_length=[sequence_length] * self.batch_size,
                                                           scope=scope)

        return outputs, output_states

    def build_attention(self, attention_type, encoder_outputs, encoder_states, rnn_type, num_layers, dim_hidden, keep_prob,
                        sequence_length, y):

        if attention_type == 0: # bahdanau
            # Attention mechanism
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=dim_hidden, memory=encoder_outputs)

        elif attention_type == 1: # luong
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=dim_hidden, memory=encoder_outputs)
        else:
            return encoder_outputs, encoder_states

        decoder_cell = self.build_rnn_cell_stack(rnn_type=rnn_type, num_layers=num_layers,
                                                 dim_hidden=dim_hidden, keep_prob=keep_prob)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                             output_attention=False)

        # Initial attention
        attention_zero = attention_cell.zero_state(batch_size=tf.shape(encoder_outputs)[0], dtype=tf.float32)
        initial_state = attention_zero.clone(cell_state=encoder_states[0])

        # Helper function
        helper = tf.contrib.seq2seq.TrainingHelper(inputs=y, sequence_length=sequence_length)
        # training_helper = TrainingHelper(inputs=self.y,  # feed in ground truth
        #                                  sequence_length=self.y_lengths)  # feed in sequence lengths

        decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_cell,
                                                  helper=helper,
                                                  initial_state=initial_state)

        decoder_outputs, decoder_final_state, decoder_final_sequence_lengths \
            = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)

        return decoder_outputs, decoder_final_state

    def build_logits(self, mode, outputs, batch_size):
        with tf.variable_scope('logits') as scope:
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

    def build_loss(self, mode, logits, batch_size):
        with tf.name_scope('loss'):
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

    def optimize(self, learning_rate, loss):
        train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # final_train_op = tf.group(train_op1, train_op2)
        return train

    def evaluate(self, mode, logits, y):
        with tf.name_scope('evaluation'):
            if mode == 0:
                pred = tf.argmax(logits, axis=2)

                # print(self.y) # Tensor("y:0", shape=(?, 10=sequence_length), dtype=int64, device=/device:CPU:0)
            elif mode == 2:
                pred = tf.argmax(logits, 1)
                # print(self.pred) # Tensor("evaluation/ArgMax:0", shape=(?,), dtype=int64)
                # print(self.y) # Tensor("y:0", shape=(?, 1), dtype=int64)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))

            correct_count = tf.reduce_sum(tf.to_float(tf.equal(pred, y)), axis=0)

        return pred, accuracy, correct_count
