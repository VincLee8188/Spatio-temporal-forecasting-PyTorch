import argparse
from os.path import join as pjoin
from models.base_model import *
from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import train_model
import random
from models.tester import *
from models.transformer import *


import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=1)
parser.add_argument('--n_his', type=int, default=36)
parser.add_argument('--n_pred', type=int, default=18)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--ks', type=int, default=2)
parser.add_argument('--rnn_units', type=int, default=64)
parser.add_argument('--n_heads', type=int, default=2)
parser.add_argument('--num_rnn_layers', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='ADAM')
parser.add_argument('--function', type=str, default='fc')
parser.add_argument('--lr_step', type=int, default=50)
parser.add_argument('--lr_gamma', type=float, default=0.5)
parser.add_argument('--use_schedule_learning', type=bool, default=1)
parser.add_argument('--cl_decay_steps', type=int, default=1)
parser.add_argument('--print_node', type=int, default=0)
parser.add_argument('--patience', type=int, default=80)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--max_grad_norm', type=float, default=1.0)


args = parser.parse_args()
print(f'Training configs: {args}')

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
ks, rnn_units, function = args.ks, args.rnn_units, args.function
num_rnn_layers, cl_decay_steps, ucl = args.num_rnn_layers, args.cl_decay_steps, args.use_schedule_learning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if n > 1:
    # Load wighted adjacency matrix W
    wa = weight_matrix(pjoin('./dataset', f'W_{n}.csv'))
    # Calculate graph kernel
    la = scaled_laplacian(wa)
    # Alternative approximation method: 1st approx - first_approx(W, n).
    lk = cheb_poly_approx(la, ks, n)
    # lk = first_approx(wa, n)
    graph_kernel = torch.tensor(lk).type(torch.float32).to(device)
else:
    # univariate time series prediction
    graph_kernel = None

# Data Preprocessing
data_file = pjoin('./dataset', f'V_{n}.csv')
n_train, n_val, n_test = 6, 1, 1
# data: [batch_size, seq_len, n_well, C_0].
dataloaders, dataset_sizes, test_data_gen, stats = loader_gen(data_file, n_train, n_val,
                                                              n_test, n_his, n_pred, n, args.batch_size, device)
print('>> Loading dataset with Mean: {:.2f}, STD: {:.2f}'.format(stats['mean'], stats['std']))

if __name__ == '__main__':
    if args.model_type == 'gcn_gru':
        model = DCRNNModel(graph_kernel, input_dim=1, output_dim=1, seq_len=n_his, horizon=n_pred, rnn_units=rnn_units,
                           num_rnn_layers=num_rnn_layers, num_nodes=n, cl_decay_steps=args.cl_decay_steps,
                           use_curriculum_learning=ucl, use_attention=use_attention, function=function).to(device)
        train_model(model, dataloaders, dataset_sizes, args, stats)
        print(args)
        print(f'The model has {count_parameters(model)} parameters')
        model_test(test_data_gen, args, stats)
    elif args.model_type == 'transformer':
        if args.function == 'fc':
            model = ShiftTransformer(rnn_units, args.n_heads, rnn_units, num_rnn_layers, dropout=args.dropout).to(device)
        elif args.function == 'gconv':
            model = GCNShiftTransformer(ks, graph_kernel, rnn_units, args.n_heads, rnn_units, num_rnn_layers,
                                        dropout=args.dropout).to(device)
        train_model(model, dataloaders, dataset_sizes, args, stats)
        print(args)
        print(f'The model has {count_parameters(model)} parameters')
        model_multi_test(test_data_gen, args, stats)
    else:
        raise ValueError(f'ERROR: no model type named {args.model_type}')

