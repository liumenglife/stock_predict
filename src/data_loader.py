from os import listdir
from os.path import join, isfile
import random
import copy
import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader:
    class DataSet:
        def __init__(self, data_path, sequence_length=5, label_term=10):
            # Current data format
            # date, final_price, compare_to_prior, start_price, highest_price, lowest_price, num_of_traded

            if sequence_length < 5:
                print('Length set is too short')
                return

            self.data_path = data_path
            self.data_list = sorted([f.replace('.npy', '') for f in listdir(self.data_path)
                                     if isfile(join(self.data_path, f)) and '.DS_Store' not in f])

            self.sequence_length = sequence_length
            self.label_term = label_term

            self.all_data_dict = dict()

            print('Loading stock data by compnay name')
            for fname in self.data_list:
                data = np.load(join(self.data_path, fname + '.npy'))

                for d in data:
                    d[0] = int(d[0].replace('.', ''))
                    for i in range(1, len(d)):
                        d[i] = int(d[i])

                self.all_data_dict[fname] = data
                # print(len(data))

                # self.queries = np.load(join(self.data_path, fname + '.npy'))
                # self.labels = np.load(join(self.data_path, fname + '.npy'))

            print('Reformat by given sequence length')
            self.queries, self.labels = self._make_dataset(sequence_length=self.sequence_length, label_term=self.label_term)

            print(len(self.queries))
            print(len(self.labels))

        def __len__(self):
            return len(self.queries)

        def __getitem__(self, idx):
            return self.queries[idx], self.labels[idx]

        def _min_max_scaler(self, data):
            ''' Min Max Normalization
            Parameters
            ----------
            data : numpy.ndarray
                input data to be normalized
                shape: [Batch size, dimension]
            Returns
            ----------
            data : numpy.ndarry
                normalized data
                shape: [Batch size, dimension]
            References
            ----------
            .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
            '''
            numerator = data - np.min(data, 0)
            denominator = np.max(data, 0) - np.min(data, 0)
            # noise term prevents the zero division
            return numerator / (denominator + 1e-7)

        def _make_dataset(self, sequence_length, label_term):
            """
            Reform the sequential data and set into self.queries and labels

            :return: queries and labels
            """

            queries = list()
            labels = list()

            # Slice range from -30 to 30 with label term
            label_dict = dict()
            label_range = [-30, 30] # min max
            num_labels = int((label_range[1] - label_range[0]) / label_term)
            prev = label_range[0]
            for i in range(num_labels):
                label_dict[i] = [prev, prev + label_term]
                prev += label_term
            label_dict[num_labels - 1][1] = label_range[1] + 1
            # label_dict[0] = [-30, 0]
            # label_dict[1] = [0, 1]
            # label_dict[2] = [1, 31]


            print('Label Range:', label_dict)

            # date, final_price, compare_to_prior, start_price, highest_price, lowest_price, num_of_traded
            # Added Features: None yet
            # Label:
            for key, values in self.all_data_dict.items():
                # sort values by date
                values = sorted(values, key=lambda t: t[0])

                # add prior day feature

                # add key as feature

                # Make sets of data in sequence_length

                for pos in range(len(values)):
                    if pos + sequence_length >= len(values):
                        # print('pos + sequence_length: %i and len(values): %i' % (pos + sequence_length, len(values)))
                        break

                    # Convert features to be int
                    v = values[pos: pos + sequence_length]

                    # Make Label
                    lp = values[pos + sequence_length]

                    # If Stock data jumps up because of some reason (ex, stock merge), continue to make next
                    if self._check_validation(v, lp, label_dict) is not True:
                        # print('pos %i IS NOT VALID' % (pos))
                        continue

                    _lp = lp
                    lp = int(float(lp[1] * 100 / v[-1][1]) - 100)
                    lf = -1
                    for idx, label in label_dict.items():
                        # if lp in range(label[0], label[1]):
                        if lp >= label[0] and lp < label[1]:
                            lf = idx
                            break

                    if lf == -1:
                        # duplicate as _check_validation already checks
                        print('company: %s lp: %i pos: %i _lp[1]: %i v[-1][1]: %i' % (key, lp, pos, _lp[1], v[-1][1]))
                        print('Label could not be found. Check the range')

                    v = self._min_max_scaler(v)
                    queries.append(v)
                    labels.append(lf)

            queries = np.array(queries)
            labels = np.array(labels)
            # labels = np.array([[np.int64(x)] for x in labels])

            return queries, labels

        def _check_validation(self, v, lp, label_dict):
            """

            :param v:
            :param lp:
            :return: Max range
            """

            is_valid = True

            for idx, value in enumerate(v):
                if idx == len(v) - 1:
                    r = int(float(lp[1] * 100 / v[-1][1]) - 100)

                else:
                    r = int(float(v[idx + 1][1] * 100 / v[idx][1]) - 100)

                lf = -1
                for idx2, label in label_dict.items():
                    # if r in range(label[0], label[1]):
                    if r >= label[0] and r < label[1]:
                        lf = idx2
                        break

                if lf == -1:
                    is_valid = False
                    return is_valid

            return is_valid

        def reshuffle(self):
            combined = list(zip(self.queries, self.labels))
            random.shuffle(combined)
            self.queries, self.labels = zip(*combined)

    def __init__(self, data_path, sequence_length=10, train_test_ratio=0.1, label_term=10):
        self.dataset = self.DataSet(data_path=data_path, sequence_length=sequence_length, label_term=label_term)

        self.label_list = np.unique(self.dataset.labels)
        for label in self.label_list:
            # print(np.where(self.dataset.labels == label))
            print('Label %i and num of data: %i' % (label, len(np.where(self.dataset.labels == label)[0])))

        x_train, x_test, y_train, y_test \
            = train_test_split(self.dataset.queries, self.dataset.labels, test_size=train_test_ratio)

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