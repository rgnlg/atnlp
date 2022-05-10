#from "Seq2Seq Model" - https://pytorch.org/tutorials/intermediate/seq2seq_translat
    # Encoder reads an inoput sequence, outputs a single vector
    # Decoder reads that vector to produce an output sequence

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PHILINE: Here we need to create our different models as classes (e.g. GRU/LSTM, hidden units, number of layers etc.)


# ------------------------------ ENCODER - ARCHITECTURES -----------------------------------
# outputs a vector and hidden sate for every word form the input sentence

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
# takes encoder output vector(s) and outputs a sequence of words to create the action sequence

# SIMPLE DECODER: using only last output of the encoder (context vector), at every step the decoder is given an input token and hidden state;
    # intital input: <SOS>, context vector

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_rnn, dropout_embedded):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout_rnn)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_embedded)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = self.dropout(output)

        # output = F.relu(output)

        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

    def initCell(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)


# ATTENTION DECODER: Calculating attention weights multiplied with the encoder output at every step

class AttnDecoder_layer_GRU(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, max_length, dropout_rnn, dropout_embedded):
        super(AttnDecoder_layer_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length) # feed-forward sttention layer (inputs: decoder's input and hidden states)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(dropout_embedded)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=dropout_rnn)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0)) # calculate a set of attention weights

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
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
        self.attn = nn.Linear(self.hidden_size * 2,
                              self.max_length)  # feed-forward sttention layer (inputs: decoder's input and hidden states)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(dropout_embedded)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, dropout=dropout_rnn)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))  # calculate a set of attention weights

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

    def initCell(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)