from utils.math_utils import z_score
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class WaterDataset(Dataset):
    def __init__(self, data, n_his):
        self.__data = data
        self.n_his = n_his

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return self.__data[idx, :self.n_his, :, :], self.__data[idx, self.n_his:, :, :]


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, C_0=1):
    """
    Gain dataset from the original time series.
    :param len_seq: int, length of the sequence.
    :param data_seq: np.ndarray, [len_seq, n_well * C_0].
    :param offset: start point to make the new dataset.
    :param n_frame: n_his + n_pred.
    :param n_well: number of the vertices on the graph.
    :param C_0: number of the channels of source data.
    :return: np.ndarray, [n_slot, n_frame, n_well, C_0].
    """
    n_slot = len_seq - n_frame + 1
    tmp_seq = np.zeros((n_slot, n_frame, n_route, C_0))
    # data_seq = data_seq[:-144, 3:4]
    for i in range(n_slot):
        sta = offset + i
        end = sta + n_frame
        tmp_seq[i, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq


def data_gen(file_path, data_config, n_route, n_frame, device):
    # Generate datasets for training, validation, and test.
    # file_path: the path of the file.
    # data_config: the portion of each set.
    # n_route: number of the vertices on the graph.
    # return: dict that contains training, validation and test data，stats.

    n_train, n_val, n_test = data_config
    r_train, r_val = float(n_train)/(n_train+n_val+n_test), float(n_val)/(n_train+n_val+n_test)

    try:
        data_seq = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        raise FileNotFoundError(f'ERROR: input file was not found in {file_path}.')

    length = data_seq.shape[0]
    data_frame = seq_gen(length, data_seq, 0, n_frame, n_route)
    num_data = data_frame.shape[0]
    seq_train = data_frame[:int(num_data*r_train), :, :, :]
    seq_val = data_frame[int(num_data*r_train):int(num_data*r_train)+int(num_data*r_val), :, :, :]
    seq_test = data_frame[int(num_data*r_train)+int(num_data*r_val):, :, :, :]

    # x_stats: dict, the stats for the training dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}

    # x_train, x_val, x_test: tensor, [len_seq, n_frame, n_route, C_0].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])
    x_train = torch.from_numpy(x_train).type(torch.float32).to(device)
    x_val = torch.from_numpy(x_val).type(torch.float32).to(device)
    x_test = torch.from_numpy(x_test).type(torch.float32).to(device)
    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    return x_data, x_stats


def loader_gen(data_file, n_train, n_val, n_test, n_his, n_pred, n_route, batch_size, device):
    # Wrap the dataset with data loaders.
    # data_file: the path of the file
    # n_train, n_val, n_test: the configuration of dataset.
    # n_his: length of source series.
    # n_pred: length of target series.
    # return: dict of dataloaders for training and validation, dict of sizes for training dataset
    # validation dataset, dataset for testing, statics for the dataset.
    # data: [batch_size, seq_len, n_well, C_0].

    data_wl, stats = data_gen(data_file, (n_train, n_val, n_test), n_route, n_his + n_pred, device)
    trainset = WaterDataset(data_wl['train'], n_his)
    validset = WaterDataset(data_wl['val'], n_his)
    testset = WaterDataset(data_wl['test'], n_his)
    train_data_gen = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    valid_data_gen = DataLoader(validset, batch_size=batch_size)
    test_data_gen = DataLoader(testset, batch_size=batch_size)
    dataset_sizes = {'train': len(train_data_gen.dataset), 'valid': len(valid_data_gen.dataset)}
    dataloaders = {'train': train_data_gen, 'valid': valid_data_gen}
    return dataloaders, dataset_sizes, test_data_gen, stats
