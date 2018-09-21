from src.data.data_loader import DataLoader
# from .models.rnn import RnnAttnModel
from src.models.rnn import RNN
from src.models.rnn_new import RNN as RNN_new
import sys
from .model_saver import ModelSaver
from os.path import join, exists
from os import makedirs
from datetime import datetime
from sklearn.metrics import classification_report


class Train(ModelSaver):
    def __init__(self):

        self.cpu_only = True
        self.gpu = 0
        self._model = None

    def get_model(self, model_name):
        if model_name == 'rnn':
            # self._model = RnnAttnModel
            model = RNN
        elif model_name == 'rnn_new':
            model = RNN_new
        else:
            print('No matching model found')
            sys.exit()

        return model

    def training(self, data_path, output_path, model_name, hparams=None):

        self._model = self.get_model(model_name)

        default_hparams = self._model.get_default_params()
        if hparams is not None:
            default_hparams.update_merge(hparams=hparams)
            hparams = default_hparams

        epochs = hparams.epochs
        batch_size = hparams.batch_size
        sequence_length = hparams.sequence_length
        train_test_ratio = hparams.train_test_ratio
        learning_rate = hparams.learning_rate

        data_loader = DataLoader(data_path=data_path, output_path=output_path, hparams=hparams)

        label_list = data_loader.label_list
        hparams.update(
            num_labels = len(label_list)
        )
        print('Label Length: %i' % (len(label_list)))

        model, sess, g = self._model_init(model=self._model, hparams=hparams)

        global_step = 0
        print_step_interval = 500
        step_time = datetime.now()

        highest_accuracy = 0
        early_stop_count = 0

        for epoch in range(epochs):
            data_loader.reshuffle()
            avg_loss = 0.0
            avg_accuracy = 0.0

            batch_iter_max = len(data_loader.dataset) / batch_size + 1
            for i, (data, labels) in enumerate(data_loader.batch_loader(data_loader.dataset, batch_size)):
                # print(labels)
                # print(data, labels)
                # attn_seq_length = [[sequence_length]] * len(data)
                attn_seq_length = [sequence_length] * len(data)
                _, loss, accuracy, logits, outputs = sess.run([model.train, model.loss, model.accuracy, model.logits,
                                                               model.outputs],
                                                              feed_dict={model.x: data, model.y: labels,
                                                                         model.dropout_keep_prob: 0.5,
                                                                         model.learning_rate: learning_rate,
                                                                         model.attn_seq_length: attn_seq_length
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

                if global_step % (print_step_interval * 10) == 0:

                    step_time = datetime.now()

                    if train_test_ratio != 0:
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
                        current_accuracy = t_avg_accuracy

                        print('[global_step-%i] duration: %is test_loss: %f accuracy: %f' % (global_step,
                                                                                             (datetime.now() - step_time).seconds,
                                                                                             t_avg_loss, t_avg_accuracy))
                    else:
                        current_accuracy = avg_accuracy

                    if highest_accuracy < current_accuracy:
                        print('Saving model...')
                        highest_accuracy = current_accuracy
                        current_accuracy = 0
                        if output_path is not None:
                            if not exists(output_path):
                                makedirs(output_path)
                        output_full_path = join(output_path, 'loss%f_acc%f_epoch%i' % (avg_loss, avg_accuracy, epoch + 1))
                        self.save_session(directory=output_full_path, global_step=global_step)

                    if current_accuracy != 0:
                        early_stop_count += 1

                    step_time = datetime.now()

            if early_stop_count > 2:
                learning_rate = learning_rate * 0.95


            if early_stop_count > 5:
                print('Early stopped !')
                break

    def test(self, model_path, model_name, data_path, hparams=None):

        self._model = self.get_model(model_name)

        default_hparams = self._model.get_default_params()
        if hparams is not None:
            default_hparams.update_merge(hparams=hparams)
            hparams = default_hparams

        batch_size = hparams.batch_size

        data_loader = DataLoader(data_path=data_path, hparams=hparams)

        label_list = data_loader.label_list
        hparams.update(
            num_labels=len(label_list)
        )

        model, sess, g = self._model_init(model=self._model, hparams=hparams, directory=model_path)

        t_avg_loss = 0.0
        t_avg_accuracy = 0.0
        t_batch_iter_max = len(data_loader.dataset) / batch_size + 1

        # added
        avg_precision = 0.0
        avg_recall = 0.0
        avg_f1 = 0.0
        avg_n_accuracy = 0.0

        cur_time = datetime.now()

        y_correct = list()
        y_pred = list()
        for t_i, (t_data, t_labels) in enumerate(data_loader.batch_loader(data_loader.dataset, batch_size)):
            accuracy, logits, loss, precision, recall, f1, n_accuracy, pred \
                = sess.run([model.accuracy, model.logits, model.loss, model.precision,
                            model.recall, model.f1, model.n_accuracy, model.pred],
                           feed_dict={model.x: t_data, model.y: t_labels, model.dropout_keep_prob: 1.0})

            t_avg_loss += float(loss)
            t_avg_accuracy += float(accuracy)
            avg_precision += float(precision)
            avg_recall += float(recall)
            avg_f1 += float(f1)
            avg_n_accuracy += float(n_accuracy)
            y_correct.extend(t_labels)
            y_pred.extend(pred)

        t_avg_loss = float(t_avg_loss / t_batch_iter_max)
        t_avg_accuracy = float(t_avg_accuracy / t_batch_iter_max)
        avg_precision = float(avg_precision / t_batch_iter_max)
        avg_recall = float(avg_recall / t_batch_iter_max)
        avg_f1 = float(avg_f1 / t_batch_iter_max)
        avg_n_accuracy = float(avg_n_accuracy / t_batch_iter_max)

        print('[Test Accuracy] duration: %is test_loss: %f accuracy: %f' % ((datetime.now() - cur_time).seconds,
                                                                            t_avg_loss, t_avg_accuracy))
        # print("Precision: %f Recall: %f f1: %f n_accuracy: %f" % (avg_precision, avg_recall, avg_f1, avg_n_accuracy))

        print(classification_report(y_correct, y_pred))
