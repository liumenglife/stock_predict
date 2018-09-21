import tensorflow as tf
from tensorflow import gfile


class ModelSaver:

    def restore_checkpoint(self, directory):
        """
        Load the saved model from cpkt

        :param directory: Path of cpkt
        :return: None
        """
        if directory is None:
            self.sess.run(tf.global_variables_initializer())
            return

        ckpt = tf.train.get_checkpoint_state(directory)
        if ckpt and gfile.Exists("%s.index" % ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            # self.sess.run(tf.global_variables_initializer()) # RNN
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
            print("ERROR!!! _restore_checkpoint none file ", directory)

    def save_session(self, directory, global_step):
        """
        Save the model

        :param directory:
        :param global_step:
        :return:
        """
        self.saver.save(self.sess, directory, global_step=global_step)
        self.last_path = directory

    def _create_saver(self):

        self.saver = tf.train.Saver()

    def session_initialize(self, graph=None, use_gpu=False):
        """
        Initializes session

        :param graph: Tensorflow graph
        :return:
        """
        # Summary
        if use_gpu is False:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            tf_config = tf.ConfigProto(allow_soft_placement=True)
            tf_config.gpu_options.allow_growth = True

        if graph is not None:
            self.sess = tf.Session(config=tf_config, graph=graph)
        else:
            self.sess = tf.Session(config=tf_config)

        return self.sess

    def _model_init(self, model, hparams, directory=None):
        """
        Initiate model with session and graph

        :param func: function to run in continuation
        :return: Model, session, graph
        """
        g = tf.Graph()
        with g.as_default():
            if hparams.use_gpu is False:
                with tf.device('/cpu:0'):
                    self.model = model(hparams=hparams)
            else:
                with tf.device('/gpu:' + str(hparams.gpu)):
                    self.model = model(hparams=hparams)

        self.sess = self.session_initialize(graph=g, use_gpu=hparams.use_gpu)
        with self.sess.as_default():
            with g.as_default():
                self._create_saver()
                self.restore_checkpoint(directory=directory)

        return self.model, self.sess, g

    def _close_session(self):
        """
        Close the session

        :return:
        """
        self.sess.close()
        tf.reset_default_graph()