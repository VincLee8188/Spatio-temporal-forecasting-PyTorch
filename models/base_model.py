from models.layer_module import *
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.function = model_kwargs.get('function', 'fc')
        self.num_nodes = int(model_kwargs.get('num_nodes', 38))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.gcn_gru_layers = nn.ModuleList([GRUCell(input_size=(1 if layer == 0 else self.rnn_units)
                                                     , hidden_size=self.rnn_units, nodes=self.num_nodes,
                                                     graph_kernel=adj_mx, func_type=self.function)
                                             for layer in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size), device=device)
        hidden_states = []
        output = inputs
        for layer_num, gcn_gru_layer in enumerate(self.gcn_gru_layers):
            next_hidden_state = gcn_gru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states, dim=0)


class Attention(nn.Module):
    def __init__(self, hidden_state_size):
        super().__init__()
        self.attn = nn.Linear(hidden_state_size * 2, hidden_state_size)
        self.v = nn.Parameter(torch.rand(hidden_state_size), requires_grad=True)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch_size, hidden_state_size]
        # encoder_outputs = [seq_len, batch_size, hidden_state_size]
        batch_size = encoder_outputs.shape[1]
        seq_len = encoder_outputs.shape[0]
        # hidden = [batch_size, seq_len, hidden_state_size]
        # encoder_outputs = [batch_size, seq_len, hidden_state_size]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # energy = [batch_size, seq_len, hidden_state_size]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        # v = [batch_size, 1, hidden_state_size]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # attention = [batch_size, seq_len]
        attention = torch.bmm(v, energy).squeeze(1)
        return F.softmax(attention, dim=1)


class DecoderModelWithAttention(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, attention, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))  # seq_len for the decoder
        self.projection_layer = nn.Linear(self.rnn_units * 2 + self.output_dim, self.output_dim)
        self.attention = attention
        self.gcn_gru_layers = nn.ModuleList([GRUCell(input_size=(1+self.rnn_units if layer == 0 else self.rnn_units)
                                                     , hidden_size=self.rnn_units, nodes=self.num_nodes,
                                                     graph_kernel=adj_mx, func_type=self.function)
                                             for layer in range(self.num_rnn_layers)])

    def forward(self, inputs, encoder_outputs, hidden_state=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size = inputs.shape[0]
        hidden_states = []
        a = self.attention(hidden_state[-1], encoder_outputs)
        # a = [batch_size, 1, seq_len]
        a = a.unsqueeze(1)
        # encoder_outputs = [seq_len, batch_size, hidden_state_size]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # weighted = [batch_size, num_nodes, hidden_state_size]
        weighted = torch.bmm(a, encoder_outputs).squeeze().reshape(batch_size, self.num_nodes, -1)
        # output = [batch_size, new_hidden_state] (new_hidden_state = num_nodes * (output_dim + hidden_state_size))
        output = torch.cat((inputs.reshape(batch_size, self.num_nodes, -1), weighted), dim=2).view(batch_size, -1)
        for layer_num, gcn_gru_layer in enumerate(self.gcn_gru_layers):
            next_hidden_state = gcn_gru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(torch.cat((output.view(-1, self.rnn_units), weighted.view(-1, self.rnn_units),
                                                     inputs.reshape(-1, self.output_dim)), dim=1))
        output = projected.view(batch_size, -1)

        return output, torch.stack(hidden_states, dim=0)


class DCRNNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.encoder_model = EncoderModel(adj_mx, **model_kwargs)
        self.attention = Attention(self.hidden_state_size)
        self.decoder_model = DecoderModelWithAttention(adj_mx, self.attention, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 200))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))

    def _compute_sampling_threshold(self, batches_seen):  # with the progress of training, teaching force ratio decrease
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder4attention(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_nodes * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
                 outputs: (seq_len, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        outputs = []
        for t in range(self.encoder_model.seq_len):
            output, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)
            outputs.append(output)

        return encoder_hidden_state, torch.stack(outputs, dim=0)

    def decoder4attention(self, base, encoder_hidden_state, encoder_outputs, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param base:
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        go_symbol = base
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol  # may use the last element of inputs

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, encoder_outputs,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                d = self._compute_sampling_threshold(batches_seen)
                if c < max(d, 0.5):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: outputs: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        encoder_hidden_state, encoder_outputs = self.encoder4attention(inputs)
        outputs = self.decoder4attention(inputs[-1], encoder_outputs, encoder_hidden_state, labels,
                                         batches_seen=batches_seen)
        return outputs
