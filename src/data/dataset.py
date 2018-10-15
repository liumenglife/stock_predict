from os.path import isfile, join
from os import listdir
import numpy as np
import random
from .preprocess import PreProcess


class DataSet:
    def __init__(self, data_path, data_status=0, output_path=None,
                 label_price=0, mode=2, sequence_length=5, label_term=10, normalize=True):
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
        self.normalize = normalize

        self.label_dict = self.get_label_dict()

        print('Loading stock data by compnay name')
        if data_status == 0:
            # Make sequence data
            self.sequence_data, self.end_data = PreProcess.make_sequence_data(input_path=data_path,
                                                                              output_path=output_path,
                                                                              sequence_length=sequence_length)
        elif data_status == 1:
            # Load sequenced data
            data_list = sorted([f for f in listdir(self.data_path)
                                if isfile(join(self.data_path, f)) and '.DS_Store' not in f])
            # print(data_list)

            self.sequence_data = None
            self.end_data = None

            for data in data_list:
                all_data = np.load(join(self.data_path, data))
                if self.sequence_data is None:
                    self.sequence_data = all_data[0].tolist()  # list of sequence data
                    self.end_data = all_data[1].tolist() # list of end data, will be transformed as label
                else:
                    self.sequence_data.extend(all_data[0].tolist())
                    self.end_data.extend(all_data[1].tolist())

        else:
            self.sequence_data = np.array(list())
            self.end_data = np.array(list())
            # print('Data status not provided !')

        print('Number of data: %i, %i' % (len(self.sequence_data), len(self.sequence_data)))

    def get_label_dict(self):
        label_dict = dict()
        label_range = [-30, 30]  # min max
        num_labels = int((label_range[1] - label_range[0]) / self.label_term)
        prev = label_range[0]
        for i in range(num_labels):
            label_dict[i] = [prev, prev + self.label_term]
            prev += self.label_term
        label_dict[num_labels - 1][1] = label_range[1] + 1

        print('Label Range:', label_dict)

        return label_dict

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):

        queries, labels = PreProcess.make_dataset(sequence_data=self.sequence_data[idx], end_data=self.end_data[idx],
                                                  output_path=None, label_price=None, label_term=self.label_term,
                                                  mode=self.mode, normalize=self.normalize, label_dict=self.label_dict)

        return queries, labels

    def reshuffle(self):
        combined = list(zip(self.sequence_data, self.end_data))
        random.shuffle(combined)
        self.sequence_data, self.end_data = zip(*combined)


# class DataSet:
#     def __init__(self, data_path, data_status=0, output_path=None,
#                  label_price=0, mode=2, sequence_length=5, label_term=10, normalize=True):
#         """
#
#         :param data_path:
#         :param data_status: data's status 0 - complete raw data, 1 - sequenced data, 2 - sequenced with label data
#         :param label_price: label classifier or actual price finder?
#         :param mode: 0 Many to many
#         :param sequence_length:
#         :param label_term:
#         """
#         # Current data format
#         # date, final_price, compare_to_prior, start_price, highest_price, lowest_price, num_of_traded
#
#         if sequence_length < 5:
#             print('Length set is too short')
#             return
#
#         self.data_path = data_path
#
#         self.mode = mode # 0 Many to many, 1 Many to one and many to many, 2 Many to one
#
#         self.sequence_length = sequence_length
#         self.label_term = label_term
#
#         print('Loading stock data by compnay name')
#         if data_status == 0:
#             # Make sequence data
#             sequence_data, end_data = PreProcess.make_sequence_data(input_path=data_path,
#                                                                     output_path=output_path,
#                                                                     sequence_length=sequence_length)
#
#             # Make targets (Y)
#             self.queries, self.labels = PreProcess.make_dataset(sequence_data=sequence_data, end_data=end_data,
#                                                                 output_path=output_path, label_price=label_price,
#                                                                 label_term=label_term, mode=mode, normalize=normalize)
#         elif data_status == 1:
#             # Load sequenced data
#             data_list = sorted([f for f in listdir(self.data_path)
#                                 if isfile(join(self.data_path, f)) and '.DS_Store' not in f])
#             # print(data_list)
#
#             sequence_data = None
#             end_data = None
#
#             for data in data_list:
#                 all_data = np.load(join(self.data_path, data))
#                 # if sequence_data is None:
#                 #     sequence_data = all_data[0]  # list of sequence data
#                 #     end_data = all_data[1]  # list of end data, will be transformed as label
#                 # else:
#                 #     sequence_data = np.concatenate([sequence_data, all_data[0]], axis=0)
#                 #     end_data = np.concatenate([end_data, all_data[0]], axis=0)
#                 if sequence_data is None:
#                     sequence_data = all_data[0].tolist()  # list of sequence data
#                     end_data = all_data[1] .tolist() # list of end data, will be transformed as label
#                 else:
#                     sequence_data.extend(all_data[0].tolist())
#                     end_data.extend(all_data[1].tolist())
#
#             # print(type(sequence_data))
#             # print(type(end_data))
#             # print(len(sequence_data))
#             # print(len(end_data))
#
#             # Make targets (Y)
#             self.queries, self.labels = PreProcess.make_dataset(sequence_data=sequence_data, end_data=end_data,
#                                                                 output_path=output_path, label_price=label_price,
#                                                                 label_term=label_term, mode=mode, normalize=normalize)
#         elif data_status == 2:
#             # all_data = np.load(data_path).tolist()
#             # self.queries = all_data['queries'] # list of input data
#             # self.labels = all_data['labels'] # list of label data
#             data_list = sorted([f for f in listdir(self.data_path)
#                                 if isfile(join(self.data_path, f)) and '.DS_Store' not in f])
#             self.queries = None
#             self.labels = None
#
#             for data_name in data_list:
#                 data = np.load(join(self.data_path, data_name)).tolist()
#                 if self.queries is None and self.labels is None:
#                     self.queries = data['queries']
#                     self.labels = data['labels']
#                 else:
#                     self.queries = np.concatenate([self.queries, data['queries']], axis=0)
#                     self.labels = np.concatenate([self.labels, data['labels']], axis=0)
#
#         else:
#             self.queries = np.array(list())
#             self.labels = np.array(list())
#
#         print('Length of queries: %i Length of labels: %i' % (len(self.queries), len(self.labels)))
#
#     def __len__(self):
#         return len(self.queries)
#
#     def __getitem__(self, idx):
#         return self.queries[idx], self.labels[idx]
#
#     def reshuffle(self):
#         combined = list(zip(self.queries, self.labels))
#         random.shuffle(combined)
#         self.queries, self.labels = zip(*combined)
