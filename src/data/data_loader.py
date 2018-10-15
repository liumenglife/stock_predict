import copy
import numpy as np
from .dataset import DataSet
from sklearn.model_selection import train_test_split


class DataLoader:

    # def __init__(self, data_path, mode, output_path=None, data_status=2, sequence_length=10, train_test_ratio=0.1,
    #              label_term=10):
    def __init__(self, data_path, test_path, hparams, output_path=None):

        data_status = hparams.data_status
        mode = hparams.mode
        sequence_length = hparams.sequence_length
        label_term = hparams.label_term
        normalize = hparams.normalize

        self.dataset = DataSet(data_path=data_path, output_path=output_path, data_status=data_status,
                               mode=mode, sequence_length=sequence_length, label_term=label_term, normalize=normalize)

        self.label_list = self.dataset.label_dict.keys()

        if data_path != test_path:
            self.test_dataset = DataSet(data_path=test_path, output_path=output_path + '_test', data_status=data_status,
                                        mode=mode, sequence_length=sequence_length, label_term=label_term,
                                        normalize=normalize)
        else: # test mode
            # self.test_dataset = DataSet(data_path=data_path, output_path=output_path, data_status=-1,
            #                             mode=mode, sequence_length=sequence_length, label_term=label_term,
            #                             normalize=normalize)
            # self.test_dataset.sequence_data = None
            # self.test_dataset.end_data = None
            #
            # x_train, x_test, y_train, y_test \
            #     = train_test_split(self.dataset.sequence_data, self.dataset.end_data, test_size=0.1)
            #
            # self.test_dataset.sequence_data = x_test
            # self.test_dataset.end_data = y_test
            #
            # self.dataset.queries = x_train
            # self.dataset.labels = y_train
            print('test mode')

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