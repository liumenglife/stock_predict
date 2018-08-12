import tensorflow as tf
import numpy as np
from .data_types import DataTypes


class RnnAttnModel(object):
    """
    RNN Attention model
    """

    def __init__(self, config=None, name='', vocab_size=10000, reuse=False):
        """

        :param config: Uses ModelConfig class as configuration
        :param name: Custom name of the model
        :param vocab_size: Size of vocabulary
        :param reuse: Reuse option of tensorflow variables
        """
        # tensorflow graph input
        self.name = name
        self.reuse = reuse
        self.config = config
        self.vocab_size = vocab_size
        self.attn_type = config.attn_type
        self.attn_last_output = config.attn_last_output

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, name='learning_rate')
        self.x = tf.placeholder(tf.int32, [None, config.seq_length], name='x')
        self.y = tf.placeholder(tf.int64, [None], name='y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.seq_len = tf.placeholder(dtype=tf.int32, shape=[None], name='seq_len')
        self.mask = tf.placeholder(dtype=tf.int64, shape=[None, config.seq_length], name='seq_mask')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        with tf.variable_scope('embedding_layer', reuse=self.reuse) :
            w = tf.get_variable('w', shape=[self.vocab_size, config.dim_emb], initializer=tf.random_uniform_initializer(-1, 1))
            x_embed = tf.nn.embedding_lookup(w, self.x)    # (batch_size, seq_length, dim_emb)
            #x_embed = tf.expand_dims(x_embed, 3)           # (batch_size, seq_length, dim_emb, 1)

        with tf.variable_scope('dynamic_rnn') as scope:
            stacked_rnn_cell = list()
            for i in range(config.num_layers):
                if config.cell_type == DataTypes.CELL_TYPE_LIST['lstm'] :
                    rnn_cell = tf.contrib.rnn.LSTMCell(num_units=config.dim_hidden)
                elif config.cell_type == DataTypes.CELL_TYPE_LIST['gru'] :
                    rnn_cell = tf.contrib.rnn.GRUCell(num_units=config.dim_hidden)
                else :
                    rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=config.dim_hidden)

                rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_prob)
                stacked_rnn_cell.append(rnn_cell)

            # rnn_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn_cell, state_is_tuple=True)
            rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_cell, state_is_tuple=True)

            if config.use_bidirectional is True :
                stacked_rnn_cell_bw = list()
                for i in range(config.num_layers):
                    if config.cell_type == DataTypes.CELL_TYPE_LIST['lstm'] :
                        rnn_cell_bw = tf.contrib.rnn.LSTMCell(num_units=config.dim_hidden)
                    elif config.cell_type == DataTypes.CELL_TYPE_LIST['gru'] :
                        rnn_cell_bw = tf.contrib.rnn.GRUCell(num_units=config.dim_hidden)
                    else :
                        rnn_cell_bw = tf.contrib.rnn.RNNCell(num_units=config.dim_hidden)

                    rnn_cell_bw = tf.contrib.rnn.DropoutWrapper(rnn_cell_bw, output_keep_prob=self.dropout_keep_prob)
                    stacked_rnn_cell_bw.append(rnn_cell_bw)

                # rnn_cell_bw = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn_cell_bw, state_is_tuple=True)
                rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_cell_bw, state_is_tuple=True)

                self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell_bw, inputs=x_embed,
                                                                  sequence_length=self.seq_len,
                                                                  dtype=tf.float32, scope=scope)
                self.outputs = tf.concat(self.outputs, 2)
                self.modify_dim_h = config.dim_hidden * 2

            else :
                self.outputs, self.states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                    inputs=x_embed, sequence_length=self.seq_len,
                                                    dtype=tf.float32, scope=scope)
                self.modify_dim_h = config.dim_hidden

            self.relevant_outputs = self.last_relevant(self.outputs, self.seq_len)

            if self.attn_type is not DataTypes.ATTENTION_TYPE_LIST['none'] :
                with tf.variable_scope('attn', reuse=self.reuse) :
                    self.w1_attn = tf.get_variable('w1', shape=[self.modify_dim_h, self.modify_dim_h],
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
                    self.w2_attn = tf.get_variable('w2', shape=[self.modify_dim_h, self.modify_dim_h],
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
                    self.w3_attn = tf.get_variable('w3', shape=[self.modify_dim_h, 1],
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
                    self.b_attn = tf.get_variable('b2', shape=[self.modify_dim_h], initializer=tf.zeros_initializer())

                    self.concat_w = tf.get_variable('concat_w', shape=[self.modify_dim_h*2, self.modify_dim_h],
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
                """
                with tf.variable_scope('prev_init_h', reuse=self.reuse) :
                    self.w_init = tf.get_variable('w', shape=[config.dim_hidden, 1],
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
                    self.b_init = tf.get_variable('b', shape=[config.dim_hidden], initializer=tf.zeros_initializer())
                """

                # 마지막 encode cell 사용
                if self.attn_last_output is True :
                    h_prev = self.relevant_outputs
                else :
                    h_prev = tf.reduce_mean(self.outputs, reduction_indices=1)

                #h_prev = tf.nn.tanh(tf.matmul(h_prev, self.w_init) + self.b_init)

                self.context, self.alpha = self.attention_mechanism(self.outputs, h_prev,
                                          self.mask,
                                          self.w1_attn,
                                          self.w2_attn,
                                          self.w3_attn,
                                          self.b_attn,
                                          self.concat_w)

                if config.ctx_type is DataTypes.CONTEXT_TYPE_LIST['only'] :
                    print('ctx_only')
                    self.relevant_outputs = self.context
                elif config.ctx_type is DataTypes.CONTEXT_TYPE_LIST['mul'] :
                    print('ctx_mul')
                    self.relevant_outputs = tf.multiply(self.context, self.relevant_outputs)
                else :
                    print('ctx_concat')
                    self.modify_dim_h = self.modify_dim_h * 2
                    self.relevant_outputs = tf.concat([self.context, self.relevant_outputs], 1)

        with tf.variable_scope('logits', reuse=self.reuse):
            self.w = tf.get_variable('w', shape=[self.modify_dim_h, config.label_size], initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable('b', shape=[config.label_size], initializer=tf.constant_initializer(0.0))
            #temp_outputs = tf.reshape(outputs[:, -1, :], [-1, dim_hidden*2])

            self.logits = tf.matmul(self.relevant_outputs, self.w) + self.b

        with tf.name_scope('loss') :
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))

        self.build_train(config)

    def last_relevant(self, outputs, length) :
        """

        :param outputs:
        :param length:
        :return:
        """
        batch_size = tf.shape(outputs)[0]
        max_length = int(outputs.get_shape()[1])
        output_size = int(outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(outputs, [-1, output_size])
        relevant = tf.gather(flat, index)

        return relevant

    def attention_mechanism(self, h_encoded, h_prev, mask, w1, w2, w3, b, concat_w):
        """
        Select adaptively the relevant part of the encoder's hidden states

        :param h_encoded:
        :param h_prev:
        :param mask:
        :param w1:
        :param w2:
        :param w3:
        :param b:
        :param concat_w:
        :return:
        """
        # [batch, step, dim]
        # h_prev: [batch, dim]
        batch_length = tf.shape(h_encoded)[0]
        source_seq_length = tf.shape(h_encoded)[1]
        dim_h = tf.shape(h_encoded)[2]
        step_size = int(h_encoded.get_shape()[1])

        #print("b", self.config.batch_size, batch_length, source_seq_length, dim_h)

        if self.attn_type is DataTypes.ATTENTION_TYPE_LIST['bah'] :
            # h_encoded: [batch*seq, dim]
            h_encoded = tf.reshape(h_encoded, shape=[-1, dim_h]) # 3dim -> 2dim
            # h_encoded: [batch, step, dim]
            h_encoded = tf.reshape(tf.matmul(h_encoded, w1), [-1, source_seq_length, dim_h]) # 2dim -> 3dim

            # [batch, seq, dim]
            # if tanh
            if self.config.act_type is DataTypes.ACTIVATION_TYPE_LIST['relu'] :
                h_calc = tf.nn.relu(h_encoded + tf.expand_dims(tf.matmul(h_prev, w2), 1) + b)  # note for broadcasting
            elif self.config.act_type is DataTypes.ACTIVATION_TYPE_LIST['tanh'] :
                h_calc = tf.nn.tanh(h_encoded + tf.expand_dims(tf.matmul(h_prev, w2), 1) + b)  # note for broadcasting

            # [batch*seq, dim]
            h_calc = tf.reshape(h_calc, [-1, dim_h])

            # [batch*seq, dim] * [dim, 1]
            # [batch, seq]
            self.score = tf.reshape(tf.matmul(h_calc, w3), [-1, source_seq_length])

        elif self.attn_type is DataTypes.ATTENTION_TYPE_LIST['dot'] :
            # [batch*seq, dim] * [batch, dim].T  => [batch*seq, batch]
            """
            for s in range(self.config.batch_size) :
                h_list.append( tf.matmul(h_encoded[s], tf.expand_dims(h_prev[s], 1)) )
            self.score = tf.reshape(h_list, [-1, source_seq_length])
            """

            # [batch, step, dim]
            # h_prev: [batch, dim]
            #h_enc_calc = tf.transpose(h_encoded, perm=[1,0,2]) # [step, batch, dim]
            h_enc_calc = tf.reshape(h_encoded, shape=[-1, dim_h]) # [batch-step,dim]

            h_enc_calc = tf.matmul(h_enc_calc, h_prev, transpose_b=True) #[batch-step,batch]
            h_enc_calc = tf.transpose(h_enc_calc, perm=[1,0]) # [batch, batch-step]

            index = (tf.range(0, batch_length) * batch_length) + tf.range(0, batch_length)
            flat = tf.reshape(h_enc_calc, shape=[-1, source_seq_length]) #[batch-batch, step]
            # [batch, step]
            self.score = tf.gather(flat, index)


            """
            h_enc_calc = tf.transpose(h_encoded, perm=[1,0,2]) # [step, batch, dim]
            h_enc_calc = tf.reshape(h_enc_calc, shape=[-1, dim_h]) # [step-batch,dim]

            h_enc_calc = tf.matmul(h_enc_calc, h_prev, transpose_b=True)
            #[step*batch, batch] => [step, batch, batch]
            h_enc_calc = tf.reshape(h_enc_calc, [source_seq_length, batch_length, batch_length])

            h_enc_calc = tf.matrix_diag_part(h_enc_calc)
            # [batch, seq]
            self.score = tf.matrix_transpose(h_enc_calc)
            """

        elif self.attn_type is DataTypes.ATTENTION_TYPE_LIST['general'] :
            # [batch, step, dim]
            h_encoded = tf.reshape(h_encoded, shape=[-1, dim_h]) # 3dim -> 2dim
            # h_encoded: [batch, step, dim]
            h_encoded = tf.reshape(tf.matmul(h_encoded, w1)+b, [-1, source_seq_length, dim_h]) # 2dim -> 3dim

            #h_enc_calc = tf.transpose(h_encoded, perm=[1,0,2]) # [step, batch, dim]
            h_enc_calc = tf.reshape(h_encoded, shape=[-1, dim_h]) # [batch-step,dim]

            h_enc_calc = tf.matmul(h_enc_calc, h_prev, transpose_b=True) #[batch-step,batch]
            h_enc_calc = tf.transpose(h_enc_calc, perm=[1, 0]) # [batch, batch-step]

            index = (tf.range(0, batch_length) * batch_length) + tf.range(0, batch_length)
            flat = tf.reshape(h_enc_calc, shape=[-1, source_seq_length]) #[batch-batch, step]
            # [batch, step]
            self.score = tf.gather(flat, index)


        elif self.attn_type is DataTypes.ATTENTION_TYPE_LIST['concat'] :
            # h_encoded: [batch*seq, dim]
            h_enc_calc = tf.transpose(h_encoded, perm=[1,0,2]) # [step, batch, dim]
            h_list = []
            dim_cat = dim_h*2
            for i in range(step_size) :
                h_list.append( tf.concat ([h_enc_calc[i], h_prev], 1) )
            h_cat = tf.stack(h_list)
            h_cat = tf.transpose(h_cat, perm=[1,0,2]) # [batch, step, dim]
            # h_encoded: [batch, step, dim]
            h_cat = tf.reshape(h_cat, shape=[-1, dim_cat]) # 3dim -> 2dim

            h_cat = tf.reshape(tf.matmul(h_cat, concat_w)+b, [-1, source_seq_length, dim_h]) # 2dim -> 3dim

            if self.config.act_type is DataTypes.ACTIVATION_TYPE_LIST['relu'] :
                h_calc = tf.reshape(tf.nn.relu(h_cat), shape=[-1, dim_h])# [batch*seq, dim]
            elif self.config.act_type is DataTypes.ACTIVATION_TYPE_LIST['tanh'] :
                h_calc = tf.reshape(tf.nn.tanh(h_cat), shape=[-1, dim_h])# [batch*seq, dim]

            # [batch*seq, dim] * [dim, 1]
            # [batch, seq]
            self.score = tf.reshape(tf.matmul(h_calc, w3), [-1, source_seq_length])


        # alpha: [batch, step]
        alpha = tf.nn.softmax(self.score * tf.cast(mask, dtype=tf.float32), name='attention_weights')
        #alpha = tf.nn.softmax(self.score * (tf.cast(mask, dtype=tf.float32)+((tf.cast(mask, dtype=tf.float32)-1)*10000)), name='attention_weights')

        alpha = alpha * tf.cast(mask, dtype=tf.float32)
        # [batch, step, 1]
        # [batch, step, dim] * [batch, step, 1]
        # [batch, step, dim]
        context = h_encoded * tf.expand_dims(alpha, dim=2)
        # [batch, dim]
        context = tf.reduce_sum(context, reduction_indices=1)

        return context, alpha

    def build_train(self, config):
        """

        :param config:
        :return:
        """
        with tf.name_scope('train_optimizer'):
            if config.grad_clip > 0 :
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), config.grad_clip)
                optimizer = tf.train.AdamOptimizer(self.lr_placeholder)
                #optimizer = tf.train.RMSPropOptimizer(self.lr_placeholder)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            else :
                self.train_op = tf.train.AdamOptimizer(
                    self.lr_placeholder).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope('evaluation'):
            self.pred = tf.arg_max(self.logits, 1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.y), tf.float32))

            self.conf_matrix = tf.confusion_matrix(self.y, self.pred, num_classes=config.label_size)
            self.correct_count = tf.reduce_sum(tf.to_float(tf.equal(self.pred, self.y)), axis=0)

        with tf.name_scope('summary'):
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            self.summary_op = tf.summary.merge_all()

    def pred_run(self, docs2idx, idx_dict):
        """

        :param docs2idx:
        :param idx_dict:
        :return:
        """
        # build mask
        mask = (docs2idx != idx_dict['<PAD>'])
        dynamic_seq_len = np.array([m[m == True].shape[0] for m in mask])

        feed_dict = {
            self.x: docs2idx,
            self.dropout_keep_prob: 1,
            self.seq_len: dynamic_seq_len,
            self.mask: mask
        }

        if self.attn_type is not DataTypes.ATTENTION_TYPE_LIST['none']:
            session_input = [
                                tf.nn.softmax(self.logits),
                                self.alpha
                            ]
        else :
            session_input = [
                                tf.nn.softmax(self.logits)
                            ]

        return feed_dict, session_input