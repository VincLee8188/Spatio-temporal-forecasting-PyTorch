import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
from utils.math_utils import mape, mae, mse
from utils.math_utils import z_inverse


def model_test(dataloader, args, stats):
    """
    Load and test saved model from the output directory for rnn.
    :param dataloader: instance of class Dataloader, dataloader for test.
    :param args: instance of class argparse, args for training.
    :param stats: dict, mean and variance for the test dataset.
    :return:
    """
    n_his, n_pred, ks, batch_size = args.n_his, args.n_pred, args.ks, args.batch_size
    n_route, epoch = args.n_route, args.epoch
    print_node = args.print_node
    model_path = pjoin('./output', f'{args.function}.pkl')

    model = torch.load(model_path)
    print(f'>> Loading saved model from {model_path} ...')

    v, v_ = [], []

    with torch.no_grad():
        for j, (x, y_tar) in enumerate(dataloader):
            x, y_tar = x.permute(1, 0, 2, 3), y_tar.permute(1, 0, 2, 3)
            x = x.reshape(n_his, -1, n_route)  # n_route * c_in if c_in not 1
            y_tar = y_tar.reshape(n_pred, -1, n_route)  # n_route * c_out if c_out not 1

            # [seq_len, batch_size, n_well * C_0]
            y_pre = model(x)
            # [batch_size, seq_len, n_well * C_0]
            v.extend(y_tar.transpose(0, 1).to('cpu').numpy())
            v_.extend(y_pre.transpose(0, 1).to('cpu').numpy())

        v = torch.from_numpy(np.array(v))
        v_ = torch.from_numpy(np.array(v_))
        # convert water level to its original value
        v = z_inverse(v, stats['mean'], stats['std'])
        v_ = z_inverse(v_, stats['mean'], stats['std'])

        mae1 = mae(v, v_)
        mape1 = mape(v, v_)
        mse1 = mse(v, v_)
        rmse1 = torch.sqrt(mse1)

        # plot comparison diagram of prediction value and actual measurement value
        x1 = torch.arange(len(v))
        for point in range(n_pred):
            fig = plt.figure()
            plt.title(f'Comparision plot between actual and prediction values', color='black')
            plt.xlabel("Number of test data")
            plt.ylabel('Water level [in meters]')
            plt.plot(x1, v[:, point, print_node], color='r', label="Target")
            plt.plot(x1, v_[:, point, print_node], color='b', label='Prediction')
            plt.legend()
            fig.savefig(pjoin('./picture', f'single_point{point}_node{print_node}.png'))

        # plot comparison diagram of prediction value and actual measurement value
        x2 = torch.arange(n_pred) + 1
        for node in range(1):
            fig = plt.figure()
            plt.title(f'Comparision plot between actual and prediction values', color='black')
            plt.xlabel('time step')
            plt.ylabel('water level [in meters]')
            plt.xlim(0, 19)
            plt.plot(x2, v[-1, :, node], color='r', label='Target')
            plt.plot(x2, v_[-1, :, node], color='b', label='Prediction')
            plt.legend()
            fig.savefig(pjoin('./picture', f'single_n_pred{n_pred}_node{node}.png'))

        print(f'Preprocess {j:3d}',
              f'mae<{mae1:.3f}> mape<{mape1:.3f}> mse<{mse1:.3f}> rmse<{rmse1:.3f}>')
    print('Testing model finished!')


def model_multi_test(dataloader, args, stats):
    """
    Load and test saved model from the output directory for transformer.
    :param dataloader: instance of class Dataloader, dataloader for test.
    :param args: instance of class argparse, args for training.
    :param stats: dict, mean and variance for the test dataset.
    :return:
    """
    n_his, n_pred, ks, batch_size = args.n_his, args.n_pred, args.ks, args.batch_size
    n_route, epoch = args.n_route, args.epoch
    print_node = args.print_node
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = pjoin('./output', f'{args.model_type}.pkl')

    model = torch.load(model_path)
    print(f'>> Loading saved model from {model_path} ...')

    v, v_ = [], []

    with torch.no_grad():
        for j, (x, y_tar) in enumerate(dataloader):
            x, y_tar = x.permute(1, 0, 2, 3), y_tar.permute(1, 0, 2, 3)
            # [seq_len, batch_size, n_well * C_0]
            x = x.reshape(n_his, -1, n_route)  # n_route * c_in
            y_tar = y_tar.reshape(n_pred, -1, n_route)  # n_route * c_out
            step_list = [x[-1]]
            for step in range(n_pred):
                trg_tensor = torch.stack(step_list, dim=0)
                y_pre = model(x, trg_tensor)
                step_list.append(y_pre[-1])

            # [batch_size, seq_len, n_well * C_0]
            v.extend(y_tar.transpose(0, 1).to('cpu').numpy())
            v_.extend(torch.stack(step_list[1:], dim=0).transpose(0, 1).to('cpu').numpy())

        v = torch.from_numpy(np.array(v))
        v_ = torch.from_numpy(np.array(v_))
        # convert water level to its original value
        v = z_inverse(v, stats['mean'], stats['std'])
        v_ = z_inverse(v_, stats['mean'], stats['std'])
        mae1 = mae(v, v_)
        mape1 = mape(v, v_)
        mse1 = mse(v, v_)
        rmse1 = torch.sqrt(mse1)

        # plot comparison diagram of prediction value and actual measurement value
        x1 = torch.arange(len(v))
        for point in range(n_pred):
            fig = plt.figure()
            plt.title(f'Comparision plot between actual and prediction values', color='black')
            plt.xlabel("Number of test data")
            plt.ylabel('Water level [in meters]')
            plt.plot(x1, v[:, point, print_node], color='r', label="Target")
            plt.plot(x1, v_[:, point, print_node], color='b', label='Prediction')
            plt.legend()
            fig.savefig(pjoin('./picture', f'multi_point{point}_node{print_node}.png'))

        # plot comparison diagram of prediction value and actual measurement value
        x2 = torch.arange(n_pred)
        for node in range(6):
            fig = plt.figure()
            plt.title(f'Comparision plot between actual and prediction values', color='black')
            plt.xlabel('time step')
            plt.ylabel('water level [in meters]')
            plt.plot(x2, v[-1, :, node], color='r', label='Target')
            plt.plot(x2, v_[-1, :, node], color='b', label='Prediction')
            plt.legend()
            fig.savefig(pjoin('./picture', f'multi_n_pred{n_pred}_node{node}.png'))

        # with SummaryWriter('./tensorboard') as w_test:
        #     w_test.add_figure('Prediction vs Target', figure)
        #     for i in np.arange(v_.shape[1]):
        #         w_test.add_scalars('Target&Prediction', {'target': v[i, 0], 'prediction': v_[i, 0]}, i + 1)

        print(f'Preprocess {j:3d}',
              f'mae<{mae1:.3f}> mape<{mape1:.3f}> mse<{mse1:.3f}> rmse<{rmse1:.3f}>')
    print('Testing model finished!')
