import copy
import numpy as np
from .dataset import DataSet
from sklearn.model_selection import train_test_split


class DataLoader:

    # def __init__(self, data_path, mode, output_path=None, data_status=2, sequence_length=10, train_test_ratio=0.1,
    #              label_term=10):
    def __init__(self, data_path, hparams, output_path=None):

        data_status = hparams.data_status
        mode = hparams.mode
        sequence_length = hparams.sequence_length
        train_test_ratio = hparams.train_test_ratio
        label_term = hparams.label_term
        normalize = hparams.normalize

        self.dataset = DataSet(data_path=data_path, output_path=output_path, data_status=data_status,
                               mode=mode, sequence_length=sequence_length, label_term=label_term, normalize=normalize)

        self.label_list = np.unique(self.dataset.labels)

        min_length = 0
        max_length = 0
        for label in self.label_list:
            # print(np.where(self.dataset.labels == label))
            if min_length == 0:
                min_length = len(np.where(self.dataset.labels == label)[0])

            if min_length > len(np.where(self.dataset.labels == label)[0]):
                min_length = len(np.where(self.dataset.labels == label)[0])

            if max_length < len(np.where(self.dataset.labels == label)[0]):
                max_length = len(np.where(self.dataset.labels == label)[0])
            print('Label %i and num of data: %i' % (label, len(np.where(self.dataset.labels == label)[0])))

        if train_test_ratio != 0:
            self.test_dataset = DataSet(data_path=data_path, output_path=output_path, data_status=-1,
                               mode=mode, sequence_length=sequence_length, label_term=label_term, normalize=normalize)
            self.test_dataset.queries = None
            self.test_dataset.labels = None
            # Augment the lacking label part
            for label in self.label_list:
                aug_length = max_length - len(np.where(self.dataset.labels == label)[0])

                aug_mult = int(max_length / len(np.where(self.dataset.labels == label)[0]))
                q = self.dataset.queries[np.where(self.dataset.labels == label)[0]]
                l = self.dataset.labels[np.where(self.dataset.labels == label)[0]]
                for n in range(aug_mult):
                    self.dataset.queries = np.concatenate([self.dataset.queries, q], axis=0)
                    self.dataset.labels = np.concatenate([self.dataset.labels, l], axis=0)

                self.dataset.queries = np.concatenate([self.dataset.queries, q[:aug_length]], axis=0)
                self.dataset.labels = np.concatenate([self.dataset.labels, l[:aug_length]], axis=0)

            print('Dataset query: %i label: %i' % (len(self.dataset.queries), len(self.dataset.labels)))

            x_train, x_test, y_train, y_test \
                = train_test_split(self.dataset.queries, self.dataset.labels, test_size=train_test_ratio)

            self.test_dataset.queries = x_test
            self.test_dataset.labels = y_test

        else:
            x_train = self.dataset.queries
            y_train = self.dataset.labels

        self.dataset.queries = x_train
        self.dataset.labels = y_train

    def reshuffle(self):
        self.dataset.reshuffle()
        # self.test_dataset.reshuffle()

    def batch_loader(self, dataset, n=1):
        """
        데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

        :param iterable: 데이터 list, 혹은 다른 포맷
        :param n: 배치 사이즈
        :return:
        """
        length = len(dataset)
        # print('Length:', length)
        # print('at batch loader length: %d' % length)
        for n_idx in range(0, length, n):
            # print('first range')
            yield dataset[n_idx:min(n_idx + n, length)]

    # @staticmethod
    # def get_default_hparams():
    #     # TODO currently not necessary at this part
    #
    #     return HParams(
    #         learning_rate=0.001,
    #         keep_prob=0.5,
    #         use_gpu=True,
    #         rnn_type='gru',
    #         mode=0,
    #         epochs=5,
    #         batch_size=100,
    #         feature_length=35,
    #         sequence_length=20,
    #         train_test_ratio=0.05,
    #         use_bidirectional=True,
    #         dim_hidden=20,
    #         label_term=30,
    #         num_layers=1,
    #         data_status=0,
    #         attention_type=1
    #     )