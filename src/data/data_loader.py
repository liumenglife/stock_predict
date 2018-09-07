import copy
import numpy as np
from .dataset import DataSet
from sklearn.model_selection import train_test_split


class DataLoader:

    def __init__(self, data_path, mode, output_path=None, data_status=2, sequence_length=10, train_test_ratio=0.1,
                 label_term=10):
        self.dataset = DataSet(data_path=data_path, output_path=output_path, data_status=data_status,
                               mode=mode, sequence_length=sequence_length, label_term=label_term)

        self.label_list = np.unique(self.dataset.labels)
        for label in self.label_list:
            # print(np.where(self.dataset.labels == label))
            print('Label %i and num of data: %i' % (label, len(np.where(self.dataset.labels == label)[0])))

        if train_test_ratio != 0:
            x_train, x_test, y_train, y_test \
                = train_test_split(self.dataset.queries, self.dataset.labels, test_size=train_test_ratio)
        else:
            x_train = self.dataset.queries
            y_train = self.dataset.labels
            x_test = None
            y_test = None

        self.test_dataset = copy.deepcopy(self.dataset)

        self.dataset.queries = x_train
        self.dataset.labels = y_train
        self.test_dataset.queries = x_test
        self.test_dataset.labels = y_test

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