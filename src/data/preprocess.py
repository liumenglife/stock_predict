from os.path import isfile, join, exists
from os import listdir, makedirs
import numpy as np
from datetime import datetime


class PreProcess:

    @staticmethod
    def make_sequence_data(input_path, output_path, sequence_length):
        """
        This will make sequence_length data with label stored data

        :param input_path:
        :param output_path:
        :param sequence_length:
        :param normalize:
        :param label_range:
        :return:
        """

        a = datetime.now()
        data_list = sorted([f.replace('.npy', '') for f in listdir(input_path)
                            if isfile(join(input_path, f)) and '.DS_Store' not in f])

        all_sequence_data = list()
        all_end_data = list()

        for fname in data_list:

            sequence_data = list()
            end_data = list()

            values = np.load(join(input_path, fname + '.npy'))

            for d in values:
                d[0] = int(d[0].replace('.', ''))
                for i in range(1, len(d)):
                    d[i] = int(d[i])

            values = sorted(values, key=lambda t: t[0])

            # add prior day feature

            # add key as feature

            # Make sets of data in sequence_length

            for pos in range(len(values)):
                if pos + sequence_length >= len(values):
                    # print('pos + sequence_length: %i and len(values): %i' % (pos + sequence_length, len(values)))
                    break

                # Values of length of sequence
                v = values[pos: pos + sequence_length]
                e = values[pos + sequence_length]
                sequence_data.append(v)
                end_data.append(e)

            all_sequence_data.extend(sequence_data)
            all_end_data.extend(end_data)

            if output_path is not None:
                n_output_path = join(output_path, 'sequence_data')
                if not exists(n_output_path):
                    makedirs(n_output_path)
                np.save(join(n_output_path, fname), [sequence_data, end_data])

        a = datetime.now() - a
        print('Time elapsed:', a)

        return all_sequence_data, all_end_data

    @staticmethod
    def make_dataset(sequence_data, end_data, output_path, label_price, label_term, mode):
        """
        Make labels related to the given sequence data

        :param sequence_data:
        :param end_data:
        :param output_path:
        :param label_price:
        :param label_term:
        :return:
        """

        a = datetime.now()

        queries = list()
        labels = list()

        # Slice range from -30 to 30 with label term
        label_dict = dict()
        label_range = [-30, 30]  # min max
        num_labels = int((label_range[1] - label_range[0]) / label_term)
        prev = label_range[0]
        for i in range(num_labels):
            label_dict[i] = [prev, prev + label_term]
            prev += label_term
        label_dict[num_labels - 1][1] = label_range[1] + 1

        print('Label Range:', label_dict)

        # date, final_price, compare_to_prior, start_price, highest_price, lowest_price, num_of_traded
        # Added Features: None yet
        # Label:
        for seq_data, e_data in zip(sequence_data, end_data):
            v = list()
            # print(len(v))
            v.extend(seq_data)
            v.append(e_data)

            # print(len(v))
            l = list()
            for idx, value in enumerate(v):
                if idx == len(v) - 1:  # length 21 means idx 20, so when idx == 19 is the last one can be compared
                    break
                else:
                    r = int(float(
                        v[idx + 1][1] * 100 / v[idx][1]) - 100)  # Increase/decrease in ratio compare to d and d+1

                lf = -1
                for idx2, label in label_dict.items():
                    if label[0] <= r < label[1]:
                        lf = idx2
                        l.append(lf)
                        break

                if lf == -1:  # Not found in proper range
                    break

            if len(l) != len(seq_data):
                continue

            seq_data = PreProcess.min_max_scaler(seq_data)

            queries.append(seq_data)
            labels.append(l)

        queries = np.array(queries)
        labels = np.array(labels)

        if mode == 2:
            labels = labels[:, -1]

        result = dict()
        result['queries'] = queries
        result['labels'] = labels
        print('Data sequence created. saving now...')
        try:
            if output_path is not None:
                n_output_path = join(output_path, 'labelled_data_' + str(mode))
                if not exists(n_output_path):
                    makedirs(n_output_path)
                np.save(join(n_output_path, 'x_y'), result)
                # np.save(join(n_output_path, 'x'), queries)
                # np.save(join(n_output_path, 'y'), labels)
        except Exception as e:
            print('Saving Error!')
            print(e)

        a = datetime.now() - a
        print('Elapsed time:', a)
        return queries, labels

    # @staticmethod
    # def make_sequence_data(input_path, output_path, sequence_length):
    #     """
    #     This will make sequence_length data with label stored data
    #
    #     :param input_path:
    #     :param output_path:
    #     :param sequence_length:
    #     :param normalize:
    #     :param label_range:
    #     :return:
    #     """
    #
    #     data_list = sorted([f.replace('.npy', '') for f in listdir(input_path)
    #                              if isfile(join(input_path, f)) and '.DS_Store' not in f])
    #
    #     all_sequence_data = list()
    #     all_end_data = list()
    #
    #     for fname in data_list:
    #
    #         sequence_data = list()
    #         end_data = list()
    #
    #         values = np.load(join(input_path, fname + '.npy'))
    #
    #         for d in values:
    #             d[0] = int(d[0].replace('.', ''))
    #             for i in range(1, len(d)):
    #                 d[i] = int(d[i])
    #
    #         values = sorted(values, key=lambda t: t[0])
    #
    #         # add prior day feature
    #
    #         # add key as feature
    #
    #         # Make sets of data in sequence_length
    #
    #         for pos in range(len(values)):
    #             if pos + sequence_length >= len(values):
    #                 # print('pos + sequence_length: %i and len(values): %i' % (pos + sequence_length, len(values)))
    #                 break
    #
    #             # Values of length of sequence
    #             v = values[pos: pos + sequence_length]
    #             e = values[pos + sequence_length]
    #             sequence_data.append(v)
    #             end_data.append(e)
    #
    #         all_sequence_data.extend(sequence_data)
    #         all_end_data.extend(end_data)
    #
    #         sequence_data = np.array(sequence_data)
    #         end_data = np.array(end_data)
    #         if output_path is not None:
    #             sd = dict()
    #             sd['sequence_data'] = sequence_data
    #             sd['end_data'] = end_data
    #             n_output_path = join(output_path, 'sequence_data')
    #             if not exists(n_output_path):
    #                 makedirs(n_output_path)
    #             np.save(join(n_output_path, fname), sd)
    #
    #     all_sequence_data = np.array(all_sequence_data)
    #     all_end_data = np.array(all_end_data)
    #
    #     return all_sequence_data, all_end_data
    #
    #
    # @staticmethod
    # def make_dataset(sequence_data, end_data, output_path, label_price, label_term, mode):
    #     """
    #     Make labels related to the given sequence data
    #
    #     :param sequence_data:
    #     :param end_data:
    #     :param output_path:
    #     :param label_price:
    #     :param label_term:
    #     :return:
    #     """
    #
    #     # queries = list()
    #     # labels = list()
    #     # queries = np.array(list())
    #     # labels = np.array(list())
    #     print('Sequence data shape:', sequence_data.shape)
    #     queries = np.empty(shape=[sequence_data.shape[1], sequence_data.shape[2]])
    #     labels = np.array(list())
    #     # labels = list()
    #
    #
    #     # Slice range from -30 to 30 with label term
    #     label_dict = dict()
    #     label_range = [-30, 30] # min max
    #     num_labels = int((label_range[1] - label_range[0]) / label_term)
    #     prev = label_range[0]
    #     for i in range(num_labels):
    #         label_dict[i] = [prev, prev + label_term]
    #         prev += label_term
    #     label_dict[num_labels - 1][1] = label_range[1] + 1
    #
    #     print('Label Range:', label_dict)
    #
    #     # date, final_price, compare_to_prior, start_price, highest_price, lowest_price, num_of_traded
    #     # Added Features: None yet
    #     # Label:
    #     for seq_data, e_data in zip(sequence_data, end_data):
    #         v = seq_data.tolist()
    #         v.append(e_data.tolist())
    #
    #         l = list()
    #         for idx, value in enumerate(v):
    #             if idx == len(v) - 1:  # length 21 means idx 20, so when idx == 19 is the last one can be compared
    #                 break
    #             else:
    #                 r = int(float(
    #                     v[idx + 1][1] * 100 / v[idx][1]) - 100)  # Increase/decrease in ratio compare to d and d+1
    #
    #             lf = -1
    #             for idx2, label in label_dict.items():
    #                 if label[0] <= r < label[1]:
    #                     lf = idx2
    #                     l.append(lf)
    #                     break
    #
    #             if lf == -1:  # Not found in proper range
    #                 break
    #
    #         if len(l) != len(seq_data):
    #             continue
    #
    #         seq_data = PreProcess.min_max_scaler(seq_data)
    #         # l = np.array(l)
    #
    #         # queries.append(seq_data)
    #         # labels.append(l)
    #         queries = np.concatenate([queries, seq_data], axis=0)
    #         labels = np.concatenate([labels, l], axis=0)
    #
    #     # queries = np.array(queries)
    #     # labels = np.array(labels)
    #     print(labels, len(labels), len(queries))
    #
    #     if mode == 2:
    #         labels = labels[:, -1]
    #
    #     result = dict()
    #     result['queries'] = queries
    #     result['labels'] = labels
    #     print('Data sequence created. saving now...')
    #     try:
    #         if output_path is not None:
    #             n_output_path = join(output_path, 'labelled_data_' + str(mode))
    #             if not exists(n_output_path):
    #                 makedirs(n_output_path)
    #             np.save(join(n_output_path, 'x_y'), result)
    #             # np.save(join(n_output_path, 'x'), queries)
    #             # np.save(join(n_output_path, 'y'), labels)
    #     except Exception as e:
    #         print('Saving Error!')
    #         print(e)
    #
    #     return queries, labels

    @staticmethod
    def min_max_scaler(data):
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