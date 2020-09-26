from utils.math_utils import mape, mae, mse
from os.path import join as pjoin
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.math_utils import z_inverse


def train_model(model, dataloaders, dataset_sizes, args, stats):
    """
    Train the base model while doing validation for parameters choosing.
    :param model:
    :param dataloaders: dict, include train dataset and validation dataset.
    :param dataset_sizes: dict, size of train dataset and validation dataset.
    :param args: instance of class argparse, args for training.
    :param stats:
    :return:
    """
    batch_size, epoch, lr, opt, n_route = args.batch_size, args.epoch, args.lr, args.opt, args.n_route
    n_his, n_pred, patience, max_grad_norm = args.n_his, args.n_pred, args.patience, args.max_grad_norm
    seq_loaders = dataloaders
    seq_sizes = dataset_sizes
    train_loss = []
    val_loss = []
    # The loss function is defined with mean squared loss.
    loss_func = nn.MSELoss()
    # Options for the optimizer.
    if opt == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr)
    elif opt == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr)
    else:
        raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')
    # The scheduler is used for dynamic learning rate.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    start_time = time.time()
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    wait = 0
    batches_seen = 0.0

    for i in range(epoch):
        if wait >= patience:
            print('Early stop of training!')
            break
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for j, (x, y_tar) in enumerate(seq_loaders[phase]):
                # x, y_tar: [batch_size, seq_len, n_well, C_0]
                x, y_tar = x.permute(1, 0, 2, 3), y_tar.permute(1, 0, 2, 3)
                x = x.reshape(n_his, -1, n_route)  # n_route * c_in if c_in not 1
                y_tar = y_tar.reshape(n_pred, -1, n_route)  # n_route * c_out if c_out not 1

                optimizer.zero_grad()

                if phase == 'train':
                    y_pre = model(x, labels=y_tar, batches_seen=batches_seen)
                else:
                    y_pre = model(x, labels=None, batches_seen=None)

                y_tar = y_tar.transpose(0, 1)
                y_pre = y_pre.transpose(0, 1)

                # y: [b, t, n * c]
                loss = loss_func(y_pre, y_tar)
                v_ = y_pre.clone().detach()
                v = y_tar.clone().detach()
                # convert water level to its original value
                v = z_inverse(v, stats['mean'], stats['std'])
                v_ = z_inverse(v_, stats['mean'], stats['std'])
                mae1 = mae(v, v_)
                mape1 = mape(v, v_)
                mse1 = mse(v, v_)
                rmse1 = torch.sqrt(mse1)
                print('.', end='')
                if j % 5 == 4:  # every 5 batches to display information
                    print(f'{phase}: Epoch {i + 1:2d}, Step {j + 1:3d}:',
                          f'mse<{mse1:.3f}> mae<{mae1:.3f}> mape<{mape1:.3f}> rmse<{rmse1:.3f}>')

                if phase == 'train':
                    batches_seen += 1
                    loss.backward()
                    optimizer.step()
                    # gradient clipping - this does it in place
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                running_loss += loss.data.item()

            epoch_loss = running_loss / np.ceil(seq_sizes[phase] / batch_size)
            if phase == 'train':
                scheduler.step()
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    wait = 0
                else:
                    wait += 1
            print()

    model.load_state_dict(best_model_wts)
    torch.save(model, pjoin('./output', f'{args.function}.pkl'))
    time_elapsed = time.time() - start_time
    fig = plt.figure()
    x1 = torch.arange(len(train_loss))
    plt.title(f'Training loss curve', color='black')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x1, train_loss, color='r', label='train')
    plt.plot(x1, val_loss, color='b', label='val')
    fig.legend()
    fig.savefig(pjoin('./loss_curve', f'Train_Val_loss.png'))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation loss: {:.4f}'.format(best_loss))
    print('Training model finished!')
