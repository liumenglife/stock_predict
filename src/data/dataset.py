from os.path import isfile, join
from os import listdir
import numpy as np
import random
from .preprocess import PreProcess


class DataSet:
    def __init__(self, data_path, data_status=0, output_path=None,
                 label_price=0, mode=2, sequence_length=5, label_term=10):
        """

        :param data_path:
        :param data_status: data's status 0 - complete raw data, 1 - sequenced data, 2 - sequenced with label data
        :param label_price: label classifier or actual price finder?
        :param mode: 0 Many to many
        :param sequence_length:
        :param label_term:
        """
        # Current data format
        # date, final_price, compare_to_prior, start_price, highest_price, lowest_price, num_of_traded

        if sequence_length < 5:
            print('Length set is too short')
            return

        self.data_path = data_path

        self.mode = mode # 0 Many to many, 1 Many to one and many to many, 2 Many to one

        self.sequence_length = sequence_length
        self.label_term = label_term

        print('Loading stock data by compnay name')
        if data_status == 0:
            # Make sequence data
            sequence_data, end_data = PreProcess.make_sequence_data(input_path=data_path,
                                                                    output_path=output_path,
                                                                    sequence_length=sequence_length)

            # Make targets (Y)
            self.queries, self.labels = PreProcess.make_dataset(sequence_data=sequence_data, end_data=end_data,
                                                                output_path=output_path, label_price=label_price,
                                                                label_term=label_term, mode=mode)
        elif data_status == 1:
            # Load sequenced data
            data_list = sorted([f for f in listdir(self.data_path)
                                     if not isfile(join(self.data_path, f)) and '.DS_Store' not in f])
            sequence_data = None
            end_data = None

            for data in data_list:
                all_data = np.load(join(self.data_path, data))
                if sequence_data is None:
                    sequence_data = all_data[0]  # list of sequence data
                    end_data = all_data[1]  # list of end data, will be transformed as label
                else:
                    sequence_data = np.concatenate([sequence_data, all_data[0]], axis=0)
                    end_data = np.concatenate([end_data, all_data[0]], axis=0)

            # Make targets (Y)
            self.queries, self.labels = PreProcess.make_dataset(sequence_data=sequence_data, end_data=end_data,
                                                                output_path=output_path, label_price=label_price,
                                                                label_term=label_term, mode=mode)
        else:
            # all_data = np.load(data_path).tolist()
            # self.queries = all_data['queries'] # list of input data
            # self.labels = all_data['labels'] # list of label data
            data_list = sorted([f for f in listdir(self.data_path)
                                if isfile(join(self.data_path, f)) and '.DS_Store' not in f])
            self.queries = None
            self.labels = None

            for data_name in data_list:
                data = np.load(join(self.data_path, data_name)).tolist()
                if self.queries is None and self.labels is None:
                    self.queries = data['queries']
                    self.labels = data['labels']
                else:
                    self.queries = np.concatenate([self.queries, data['queries']], axis=0)
                    self.labels = np.concatenate([self.labels, data['labels']], axis=0)


        print(len(self.queries))
        print(len(self.labels))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx]

    def reshuffle(self):
        combined = list(zip(self.queries, self.labels))
        random.shuffle(combined)
        self.queries, self.labels = zip(*combined)



# class DataSet_old:
#     def __init__(self, data_path, mode=2, sequence_length=5, label_term=10):
#         # Current data format
#         # date, final_price, compare_to_prior, start_price, highest_price, lowest_price, num_of_traded
#
#         if sequence_length < 5:
#             print('Length set is too short')
#             return
#
#         self.data_path = data_path
#         self.data_list = sorted([f.replace('.npy', '') for f in listdir(self.data_path)
#                                  if isfile(join(self.data_path, f)) and '.DS_Store' not in f])
#
#         self.mode = mode # 0 Many to many, 1 Many to one and many to many, 2 Many to one
#
#         self.sequence_length = sequence_length
#         self.label_term = label_term
#
#         self.all_data_dict = dict()
#
#         print('Loading stock data by compnay name')
#         for fname in self.data_list:
#             data = np.load(join(self.data_path, fname + '.npy'))
#
#             for d in data:
#                 d[0] = int(d[0].replace('.', ''))
#                 for i in range(1, len(d)):
#                     d[i] = int(d[i])
#
#             self.all_data_dict[fname] = data
#             # print(len(data))
#
#             # self.queries = np.load(join(self.data_path, fname + '.npy'))
#             # self.labels = np.load(join(self.data_path, fname + '.npy'))
#
#         print('Reformat by given sequence length')
#         self.queries, self.labels = self._make_dataset(sequence_length=self.sequence_length, label_term=self.label_term)
#         print(self.labels)
#         print(type(self.labels))
#         print(self.labels[0])
#         print(type(self.labels[0]))
#         if self.mode == 2:
#             self.labels = self.labels[:, -1]
#
#         print(len(self.queries))
#         print(len(self.labels))
#
#     def __len__(self):
#         return len(self.queries)
#
#     def __getitem__(self, idx):
#         return self.queries[idx], self.labels[idx]
#
#     def _min_max_scaler(self, data):
#         ''' Min Max Normalization
#         Parameters
#         ----------
#         data : numpy.ndarray
#             input data to be normalized
#             shape: [Batch size, dimension]
#         Returns
#         ----------
#         data : numpy.ndarry
#             normalized data
#             shape: [Batch size, dimension]
#         References
#         ----------
#         .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
#         '''
#         numerator = data - np.min(data, 0)
#         denominator = np.max(data, 0) - np.min(data, 0)
#         # noise term prevents the zero division
#         return numerator / (denominator + 1e-7)
#
#     def _make_dataset(self, sequence_length, label_term):
#         """
#         Reform the sequential data and set into self.queries and labels
#
#         :return: queries and labels
#         """
#
#         queries = list()
#         labels = list()
#
#         # Slice range from -30 to 30 with label term
#         label_dict = dict()
#         label_range = [-30, 30] # min max
#         num_labels = int((label_range[1] - label_range[0]) / label_term)
#         prev = label_range[0]
#         for i in range(num_labels):
#             label_dict[i] = [prev, prev + label_term]
#             prev += label_term
#         label_dict[num_labels - 1][1] = label_range[1] + 1
#         # label_dict[0] = [-30, 0]
#         # label_dict[1] = [0, 1]
#         # label_dict[2] = [1, 31]
#
#
#         print('Label Range:', label_dict)
#
#         # date, final_price, compare_to_prior, start_price, highest_price, lowest_price, num_of_traded
#         # Added Features: None yet
#         # Label:
#         for key, values in self.all_data_dict.items():
#             # sort values by date
#             values = sorted(values, key=lambda t: t[0])
#
#             # add prior day feature
#
#             # add key as feature
#
#             # Make sets of data in sequence_length
#
#             for pos in range(len(values)):
#                 if pos + sequence_length >= len(values):
#                     # print('pos + sequence_length: %i and len(values): %i' % (pos + sequence_length, len(values)))
#                     break
#
#                 # Values of length of sequence
#                 v = values[pos: pos + sequence_length + 1]
#
#                 # If Stock data jumps up because of some reason (ex, stock merge), continue to make next
#                 # if self._check_validation(v, label_dict) is not True:
#                 #     # print('pos %i IS NOT VALID' % (pos))
#                 #     continue
#                 l = list()
#                 for idx, value in enumerate(v):
#                     if idx == len(v) - 1:  # length 21 means idx 20, so when idx == 19 is the last one can be compared
#                         break
#                     else:
#                         r = int(float(
#                             v[idx + 1][1] * 100 / v[idx][1]) - 100)  # Increase/decrease in ratio compare to d and d+1
#
#                     lf = -1
#                     for idx2, label in label_dict.items():
#                         if label[0] <= r < label[1]:
#                             lf = idx2
#                             l.append(lf)
#                             break
#
#                     if lf == -1:  # Not found in proper range
#                         break
#
#                 if len(l) != sequence_length:
#                     continue
#
#                 v = v[:-1]
#                 v = self._min_max_scaler(v)
#                 queries.append(v)
#                 labels.append(l)
#
#         queries = np.array(queries)
#         labels = np.array(labels)
#         # labels = np.array([[np.int64(x)] for x in labels])
#
#         return queries, labels
#
#     def reshuffle(self):
#         combined = list(zip(self.queries, self.labels))
#         random.shuffle(combined)
#         self.queries, self.labels = zip(*combined)