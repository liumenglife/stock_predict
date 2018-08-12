from .data_loader import DataLoader
import tensorflow as tf
# from .models.rnn import RnnAttnModel
from .rnn import RNN
import sys


class Train:
    def __init__(self, model_name):

        if model_name == 'rnn':
            # self._model = RnnAttnModel
            self._model = RNN
        else:
            print('No matching model found')
            sys.exit()

    def training(self, data_path, epochs=10, batch_size=50, sequence_length=10, train_test_ratio=0.1, label_term=10,
                 use_bidirectional=False, dim_hidden=10, learning_rate=0.01, num_layers=1):
        data_loader = DataLoader(data_path=data_path, sequence_length=sequence_length,
                                 train_test_ratio=train_test_ratio, label_term=label_term)

        label_list = data_loader.label_list
        print('Label Length: %i' % (len(label_list)))

        model = self._model(num_layers=num_layers, sequence_length=sequence_length, use_bidirectional=use_bidirectional,
                            dim_hidden=dim_hidden, num_labels=len(label_list))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        # sess = tf.InteractiveSession()
        # tf.global_variables_initializer().run()

        for epoch in range(epochs):
            data_loader.reshuffle()
            avg_loss = 0.0
            avg_accuracy = 0.0

            batch_iter_max = len(data_loader.dataset) / batch_size + 1
            for i, (data, labels) in enumerate(data_loader.batch_loader(data_loader.dataset, batch_size)):
                # print(labels)
                # print(data, labels)
                _, loss, accuracy, logits, outputs = sess.run([model.train, model.loss, model.accuracy, model.logits,
                                                      model.outputs],
                                             feed_dict={model.x: data, model.y: labels, model.dropout_keep_prob: 0.7,
                                                        # model.batch_size: batch_size
                                                        model.learning_rate: learning_rate
                                                        })
                # print('Batch : ', i + 1, '/', one_batch_size,
                #       ', BCE in this minibatch: ', float(loss))
                avg_loss += float(loss)
                avg_accuracy += float(accuracy)

                # if i != 0 and i % 1 == 0:
                #     print('[%i] loss: %f accuracy: %f' % (i, loss, accuracy))
                #     print('')
                #     print(labels)
                #     print(data, labels)
                #     print('')
                #     print('logits')
                #     print(logits)
                #     print('')
                #     print('outputs[:, -1]')
                #     print(outputs[:, -1])
                #     print('')
                #     print('outputs[-1:][0]')
                #     print(outputs[-1:][0])
                #     print('')
                #     print('outputs')
                #     print(outputs)
                #
                #     break

            print('[epoch] %i train_loss: %f accuracy: %f' % (epoch, float(avg_loss / batch_iter_max),
                                                              float(avg_accuracy / batch_iter_max)))

            if epoch == epochs - 1:
                print('')
                print('logits')
                print(logits)
                print('')
                print('outputs[:, -1]')
                print(outputs[:, -1])
                print('')
                print('outputs[-1:][0]')
                print(outputs[-1:][0])
                print('')
                print('outputs')
                print(outputs)

                break

    def test(self):
        pass

