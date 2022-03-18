'''
Created on Nov, 2018

@author: hugo

'''
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from ..utils.generic_utils import to_cuda


class GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(GatedFusion, self).__init__()
        '''GatedFusion module'''
        self.fc_z = nn.Linear(4 * hidden_size, hidden_size, bias=True)

    def forward(self, h_state, input):
        z = torch.sigmoid(self.fc_z(torch.cat([h_state, input, h_state * input, h_state - input], -1)))
        h_state = (1 - z) * h_state + z * input
        return h_state

class GRUStep(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(GRUStep, self).__init__()
        '''GRU module'''
        self.linear_z = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_r = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_t = nn.Linear(hidden_size + input_size, hidden_size, bias=False)

    def forward(self, h_state, input):
        z = torch.sigmoid(self.linear_z(torch.cat([h_state, input], -1)))
        r = torch.sigmoid(self.linear_r(torch.cat([h_state, input], -1)))
        t = torch.tanh(self.linear_t(torch.cat([r * h_state, input], -1)))
        h_state = (1 - z) * h_state + z * t
        return h_state

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, \
        bidirectional=False, rnn_type='lstm', rnn_dropout=None, rnn_input_dropout=None, device=None):
        super(EncoderRNN, self).__init__()
        if not rnn_type in ('lstm', 'gru'):
            raise RuntimeError('rnn_type is expected to be lstm or gru, got {}'.format(rnn_type))
        if bidirectional:
            print('[ Using bidirectional {} encoder ]'.format(rnn_type))
        else:
            print('[ Using {} encoder ]'.format(rnn_type))
        if bidirectional and hidden_size % 2 != 0:
            raise RuntimeError('hidden_size is expected to be even in the bidirectional mode!')
        self.rnn_type = rnn_type
        self.rnn_dropout = rnn_dropout
        self.rnn_input_dropout = rnn_input_dropout
        self.device = device
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_directions = 2 if bidirectional else 1
        model = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.model = model(input_size, self.hidden_size, 1, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """x: [batch_size * max_length * emb_dim]
           x_len: [batch_size]
        """
        x = dropout(x, self.rnn_input_dropout, shared_axes=[-2], training=self.training)
        sorted_x_len, indx = torch.sort(x_len, 0, descending=True)
        x = pack_padded_sequence(x[indx], sorted_x_len.data.tolist(), batch_first=True)

        h0 = to_cuda(torch.zeros(self.num_directions, x_len.size(0), self.hidden_size), self.device)
        if self.rnn_type == 'lstm':
            c0 = to_cuda(torch.zeros(self.num_directions, x_len.size(0), self.hidden_size), self.device)
            packed_h, (packed_h_t, _) = self.model(x, (h0, c0))
            packed_h_t = torch.cat([packed_h_t[i] for i in range(packed_h_t.size(0))], -1)
        else:
            packed_h, packed_h_t = self.model(x, h0)
            packed_h_t = packed_h_t.transpose(0, 1).contiguous().view(x_len.size(0), -1)

        hh, _ = pad_packed_sequence(packed_h, batch_first=True)

        # restore the sorting
        _, inverse_indx = torch.sort(indx, 0)
        restore_hh = hh[inverse_indx]
        restore_packed_h_t = packed_h_t[inverse_indx]
        restore_hh = dropout(restore_hh, self.rnn_dropout, shared_axes=[-2], training=self.training)
        restore_packed_h_t = dropout(restore_packed_h_t, self.rnn_dropout, training=self.training)
        return restore_hh, restore_packed_h_t


def dropout(x, drop_prob, shared_axes=[], training=False):
    """
    Apply dropout to input tensor.
    Parameters
    ----------
    input_tensor: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)``
    Returns
    -------
    output: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)`` with dropout applied.
    """
    if drop_prob == 0 or drop_prob == None or (not training):
        return x

    sz = list(x.size())
    for i in shared_axes:
        sz[i] = 1
    mask = x.new(*sz).bernoulli_(1. - drop_prob).div_(1. - drop_prob)
    mask = mask.expand_as(x)
    return x * mask
