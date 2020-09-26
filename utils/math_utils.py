import numpy as np
import torch


def z_score(x, mean, std):
    # Z-score normalization function: $z = (X - \mu) / \sigma $,
    # where z is the z-score, X is the value of the element,
    # $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    # x: np.ndarray, input array to be normalized.
    # mean: float, the value of mean.
    # std: float, the value of standard deviation.
    # return: np.ndarray, z-score normalized array.
    return (x - mean) / std


def z_inverse(x, mean, std):
    # The inverse of function z_score().
    # x: np.ndarray, input to be recovered.
    # mean: float, the value of mean.
    # std: float, the value of standard deviation.
    # return: np.ndarray, z-score inverse array.

    return x * std + mean


def mape(v, v_):
    # Mean absolute percentage error.
    # v: np.ndarray or int, ground truth.
    # v_: np.ndarray or int, prediction.
    # return: int, mape averages on all elements of input.

    return torch.mean(torch.abs((v_ - v) / (v + 1e-5)))


def mse(v, v_):
    # Mean squared error.
    # v: np.ndarray or int, ground truth.
    # v_: np.ndarray or int, prediction.
    # return: int, mse averages on all elements of input.

    return torch.mean((v_ - v) ** 2)


def rmse(v, v_):
    # Root mean squared error.
    # v: np.ndarray or int, ground truth.
    # v_: np.ndarray or int, prediction.
    # return: int, rmse averages on all elements of input.

    return torch.sqrt(torch.mean((v_ - v) ** 2))


def mae(v, v_):
    # Mean absolute error.
    # v: np.ndarray or int, ground truth.
    # v_: np.ndarray or int, prediction.
    # return: int, mae averages on all elements of input.

    return torch.mean(torch.abs(v_ - v))


def evaluation(y, y_, x_stats):
    # Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    # Extended version: multi-step prediction can be calculated by self-calling.
    # y: np.ndarray or int, ground truth.
    # y_: np.ndarray or int, prediction.
    # x_stats: dict, paras of z-scores (mean & std).
    # return: np.ndarray, averaged metric values.
    dim = len(y_.shape)

    if dim == 3:
        # single_step case
        v = z_inverse(y, x_stats['mean'], x_stats['std'])
        v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
        return np.array([mape(v, v_), mae(v, v_), mse(v, v_)])
    else:
        # multi_step case
        tmp_list = []
        # recursively call
        for i in range(y_.shape[0]):
            tmp_res = evaluation(y[i], y_[i], x_stats)
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=-1)
