import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, graph_kernel, nodes, func_type='fc'):
        """

        :param input_size:
        :param hidden_size:
        :param graph_kernel: tensor, (nodes, ks * nodes).
        :param nodes: int, number of nodes in the graph.
        :param func_type:
        """
        super(GRUCell, self).__init__()
        self.type = func_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nodes = nodes
        if func_type == 'gconv':
            self.ks = graph_kernel.shape[1] // self.nodes
            self.graph_kernel = graph_kernel
            self.w_ih = nn.Parameter(torch.randn(3 * hidden_size, self.ks * input_size), requires_grad=True)
            self.w_hh = nn.Parameter(torch.randn(3 * hidden_size, self.ks * hidden_size), requires_grad=True)
        elif func_type == 'fc':
            self.w_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size), requires_grad=True)
            self.w_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size), requires_grad=True)
        else:
            raise ValueError(f'ERROR: no function type named {func_type}')
        self.b_ih = nn.Parameter(torch.randn(3 * hidden_size), requires_grad=True)
        self.b_hh = nn.Parameter(torch.randn(3 * hidden_size), requires_grad=True)
        self._reset_param()

    def _reset_param(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -std, std)

    def gc(self, x, w, b):
        batch_size = x.shape[0] // self.nodes
        x = x.reshape(batch_size, self.nodes, -1)
        x_ker = torch.matmul(x.transpose(1, 2), self.graph_kernel)
        x_ker = x_ker.reshape(batch_size, -1, self.nodes).transpose(1, 2)
        return (torch.matmul(x_ker, w.t()) + b).reshape(batch_size * self.nodes, -1)

    def forward(self, x, hx):
        batch_size = x.shape[0]
        x = x.reshape(batch_size * self.nodes, -1)
        hx = hx.reshape(batch_size * self.nodes, -1)
        if self.type == 'fc':
            gi = F.linear(x, self.w_ih, self.b_ih)
            gh = F.linear(hx, self.w_hh, self.b_hh)
        else:
            gi = self.gc(x, self.w_ih, self.b_ih)
            gh = self.gc(hx, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_i + h_i)
        n = torch.tanh(i_n + r * h_n)
        return (z * hx + (1 - z) * n).reshape(batch_size, -1)
