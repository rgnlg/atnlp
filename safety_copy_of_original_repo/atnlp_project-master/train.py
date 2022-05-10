#from "Training" - https://pytorch.org/tutorials/intermediate/seq2seq_translat

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from evaluate import *

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

# -------------------------- Prepare Training Data -----------------------------
    # Input tensor (indexes of the words in the input sentence)
    # target tensor (indexes of the words in the target sentence),
    # appen <EOS> to both sequences

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# -------------------------- Training the Model -----------------------------
    # run the input sentence through the encoder, and keep track of every output and the latest hidden state

    # PHILINE: Half of the training time, here it is applied randomly- same for our task?
    # teacher-forcing = using the real target outputs as each next input, instead of using the decoder's guess as the next input
teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, iter, max_length, clip_norm_value=5.0):

    if 'LSTM' in type(encoder).__name__:
        encoder_h = encoder.initHidden()
        encoder_c = encoder.initCell()
        encoder_hidden = [encoder_h, encoder_c]
    else:
        encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = True if iter < 100000 * 0.5 else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):

            if 'Attn' in type(decoder).__name__:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):

            # decoder_output, decoder_hidden, decoder_attention = decoder(
            #     decoder_input, decoder_hidden, encoder_outputs)

            if 'Attn' in type(decoder).__name__:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_norm_value)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_norm_value)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

        # This is a helper function to print time elapsed and estimated time remaining given the current time and progress %.

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# The whole training process looks like this:
# 1) Start a timer
# 2) Initialize optimizers and criterion
# 3) Create set of training pairs
# 4) Start empty losses array for plotting

def trainIters(encoder, decoder, input_lang, output_lang, n_iters, max_length, learning_rate, clip_norm_value, pairs, test_pairs, plot_acc_file_path, plot_loss_file_path, result_file_path, print_every=1000, plot_every=500, is_experiment2=False):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    input_max_length = max_length[0]
    output_max_length = max_length[1]

    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate)
    #training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs))
    #                  for i in range(n_iters)]
    training_pairs = [tensorsFromPair(input_lang, output_lang, pair)
                      for pair in pairs]
    criterion = nn.NLLLoss()

    accs = []
    acc_lens = []

    for iter in range(1, n_iters + 1):

        encoder.train()
        decoder.train()

        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, iter, input_max_length, clip_norm_value)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            encoder.eval()
            decoder.eval()

            acc = evaluateTest(encoder, decoder, input_lang, output_lang, test_pairs, max_length)
            accs.append(acc)
            if is_experiment2:
                acc_len = evaluateTestLength(encoder, decoder, input_lang, output_lang, test_pairs, max_length)
                acc_lens.append(acc_len)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    if is_experiment2:
        showAccLenPlot(accs, acc_lens, print_every, plot_acc_file_path)
    else:
        showAccPlot(accs, print_every, plot_acc_file_path)

    showPlot(plot_losses, plot_every, plot_loss_file_path)

    _dict = {'iteration':[i for i in range(1, n_iters+1, print_every)], 'loss':plot_losses, 'acc_test': accs}

    pd.DataFrame(_dict).to_csv(result_file_path, index=False)

# -------------------------- PLOTTING RESULTS -----------------------------

# PHILINE: Alternative using tensorboard
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showAccPlot(points, plot_every, plot_file_path):
    # plt.plot(range(0, int(100000/plot_every), plot_every), points)
    print (points)
    plt.plot(range(0, 100000, plot_every), points)
    plt.xlabel('Step')
    plt.ylabel('Acc')
    plt.title('Evaluation Acc')
    plt.savefig(plot_file_path)
    plt.clf()

def showAccLenPlot(points1, points2, plot_every, plot_file_path):
    # plt.plot(range(0, int(100000/plot_every), plot_every), points1, color='r', label="original")
    # plt.plot(range(0, int(100000/plot_every), plot_every), points2, color='b', label="with given lengths")

    plt.plot(range(0, 100000, plot_every), points1, color='r', label="original")
    plt.plot(range(0, 100000, plot_every), points2, color='b', label="with given lengths")
    plt.xlabel('Step')
    plt.ylabel('Acc')
    plt.title('Evaluation Acc')
    plt.legend()
    plt.savefig(plot_file_path)
    plt.clf()

def showPlot(points, plot_every, plot_file_path):
    # plt.plot(range(0, int(100000/plot_every), plot_every), points)

    plt.plot(range(0, 100000, plot_every), points)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(plot_file_path)
    plt.clf()

    # plt.figure()
    # fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.2)
    # ax.yaxis.set_major_locator(loc)
    # plt.plot(points)