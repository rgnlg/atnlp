"""
Authors: Duo Yang (vdb927)
Olga Iarygina (hwk263)
Philine Zeinert (vdp117)

Re-Implementation of:
Lake, B. M. and Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. International Conference on Machine Learning (ICML)
(ATNLP course, Copenhagen University, 2021/22)

Date: 28/01/2022

code adapted from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------ ENCODER - ARCHITECTURES -----------------------------------
"""
    Encoder models used for Experiment 1, 2 and 3.
    For further details refer to the paper and tutorial in the header.
"""

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rnn, dropout_embedded):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_embedded)

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout_rnn)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = self.dropout(embedded)
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

    def initCell(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)


class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rnn, dropout_embedded):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_embedded)

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout_rnn)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = self.dropout(embedded)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)


# ------------------------------ DECODER -----------------------------------
"""
    Decoder models used for Experiment 1, 2 and 3.
    For further details refer to the paper and tutorial in the header.
"""

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_rnn, dropout_embedded):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout_rnn)
        self.out1 = nn.Linear(hidden_size, output_size)
        self.out2 = nn.Linear(hidden_size*num_layers, output_size)
        self.dropout = nn.Dropout(dropout_embedded)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = self.dropout(output)

        output = F.relu(output)

        output, hidden = self.lstm(output, hidden)

        #output = self.softmax(self.out1(output[0]))
        output = self.softmax(self.out2(torch.reshape(hidden[0],(1,-1))))
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

    def initCell(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)


# ATTENTION DECODERs:

class AttnDecoder_layer_GRU(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, max_length, dropout_rnn, dropout_embedded):
        super(AttnDecoder_layer_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(dropout_embedded)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=dropout_rnn)
        self.out1 = nn.Linear(self.hidden_size, self.output_size)
        self.out2 = nn.Linear(self.hidden_size*2, self.output_size)

        self.v = torch.nn.Parameter(torch.rand(self.hidden_size))
        self.W = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.U = torch.nn.Linear(self.hidden_size, self.hidden_size)

    def _get_weights(self, query, values):
        query = query.repeat(values.size(0), 1)
        weights = self.W(query) + self.U(values)
        return torch.tanh(weights) @ self.v

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)



        attn_weights = F.softmax(
            self._get_weights(hidden.squeeze(), encoder_outputs).unsqueeze(0), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        # 1)
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # output = F.log_softmax(self.out1(output[0]), dim=1)

        # 2)
        output = torch.cat((hidden.squeeze(0), attn_applied.squeeze(0)), 1)
        output = F.log_softmax(self.out2(output), dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)


class AttnDecoder_layer_LSTM(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, max_length, dropout_rnn, dropout_embedded):
        super(AttnDecoder_layer_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(dropout_embedded)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, dropout=dropout_rnn)
        self.out1 = nn.Linear(self.hidden_size, self.output_size)
        self.out2 = nn.Linear(self.hidden_size*2, self.output_size)

        self.v = torch.nn.Parameter(torch.rand(self.hidden_size))
        self.W = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.U = torch.nn.Linear(self.hidden_size, self.hidden_size)

    def _get_weights(self, query, values):
        query = query.repeat(values.size(0), 1)
        weights = self.W(query) + self.U(values)
        return torch.tanh(weights) @ self.v

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self._get_weights(hidden[0].squeeze(), encoder_outputs).unsqueeze(0), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = torch.cat((hidden[0].squeeze(0), attn_applied.squeeze(0)), 1)
        output = F.log_softmax(self.out2(output), dim=1)

        # output = F.log_softmax(self.out1(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

    def initCell(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
