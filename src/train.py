from src.data.data_loader import DataLoader
# from .models.rnn import RnnAttnModel
from .rnn import RNN
from .rnn_new import RNN as RNN_new
import sys
from .model_saver import ModelSaver
from os.path import join, exists
from os import makedirs
from datetime import datetime


class Train(ModelSaver):
    def __init__(self, model_name):

        self.cpu_only = True
        self.gpu = 0

        if model_name == 'rnn':
            # self._model = RnnAttnModel
            self._model = RNN
        elif model_name == 'rnn_new':
            self._model = RNN_new
        else:
            print('No matching model found')
            sys.exit()

    def training(self, data_path, output_path, use_gpu=False, rnn_type='gru', mode=2, epochs=10, batch_size=50,
                 sequence_length=10, train_test_ratio=0.1, label_term=10,
                 use_bidirectional=False, dim_hidden=10, learning_rate=0.01, num_layers=1, data_status=0,
                 feature_length=7, attention_type=0):

        self.cpu_only = not use_gpu
        # print(self.cpu_only)

        data_loader = DataLoader(data_path=data_path, mode=mode, sequence_length=sequence_length,
                                 train_test_ratio=train_test_ratio, label_term=label_term, output_path=output_path,
                                 data_status=data_status)

        label_list = data_loader.label_list
        print('Label Length: %i' % (len(label_list)))

        model, sess, g = self._model_init(model=self._model, gpu=self.gpu, mode=mode, num_layers=num_layers,
                                          rnn_type=rnn_type, feature_length=feature_length,
                                          attention_type=attention_type,
                                          sequence_length=sequence_length, use_bidirectional=use_bidirectional,
                                          dim_hidden=dim_hidden, num_labels=len(label_list))

        global_step = 0
        print_step_interval = 500
        step_time = datetime.now()

        for epoch in range(epochs):
            data_loader.reshuffle()
            avg_loss = 0.0
            avg_accuracy = 0.0

            cur_time = datetime.now()

            batch_iter_max = len(data_loader.dataset) / batch_size + 1
            for i, (data, labels) in enumerate(data_loader.batch_loader(data_loader.dataset, batch_size)):
                # print(labels)
                # print(data, labels)
                _, loss, accuracy, logits, outputs = sess.run([model.train, model.loss, model.accuracy, model.logits,
                                                               model.outputs],
                                                              feed_dict={model.x: data, model.y: labels,
                                                                         model.dropout_keep_prob: 0.5,
                                                                         model.learning_rate: learning_rate
                                                                         })

                avg_loss += float(loss)
                avg_accuracy += float(accuracy)
                global_step += 1

                if global_step % print_step_interval == 0:
                    print('[global_step-%i] duration: %is train_loss: %f accuracy: %f' % (
                        global_step, (datetime.now() - step_time).seconds,
                        float(avg_loss / (i + 1)),
                        float(avg_accuracy / (i + 1))))
                    step_time = datetime.now()

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

            # print('[epoch-%i] duration: %is train_loss: %f accuracy: %f' % (epoch, (datetime.now() - cur_time).seconds,
            #                                                               float(avg_loss / batch_iter_max),
            #                                                               float(avg_accuracy / batch_iter_max)))

                if global_step % (print_step_interval * 10) == 0:
                    t_avg_loss = 0.0
                    t_avg_accuracy = 0.0
                    t_batch_iter_max = len(data_loader.test_dataset) / batch_size + 1

                    for t_i, (t_data, t_labels) in enumerate(data_loader.batch_loader(data_loader.test_dataset, batch_size)):
                        accuracy, logits, loss = sess.run([model.accuracy, model.logits, model.loss],
                                                          feed_dict={model.x: t_data, model.y: t_labels,
                                                                     model.dropout_keep_prob: 1.0})

                        t_avg_loss += float(loss)
                        t_avg_accuracy += float(accuracy)

                    t_avg_loss = float(t_avg_loss / t_batch_iter_max)
                    t_avg_accuracy = float(t_avg_accuracy / t_batch_iter_max)

                    print('[global_step-%i] duration: %is test_loss: %f accuracy: %f' % (global_step,
                                                                                         (datetime.now() - cur_time).seconds,
                                                                                         t_avg_loss, t_avg_accuracy))

            if epoch == epochs - 1:
                # print('')
                # print('logits')
                # print(logits)
                # print('')
                # print('outputs[:, -1]')
                # print(outputs[:, -1])
                # print('')
                # print('outputs[-1:][0]')
                # print(outputs[-1:][0])
                # print('')
                # print('outputs')
                # print(outputs)
                # break
                if output_path is not None:
                    if not exists(output_path):
                        makedirs(output_path)
                output_full_path = join(output_path, 'loss%f_acc%f_epoch%i' % (avg_loss, avg_accuracy, epoch + 1))
                self.save_session(directory=output_full_path, global_step=global_step)

    def test(self):
        pass

