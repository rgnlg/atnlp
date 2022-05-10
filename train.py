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
import pandas as pd
import torch.nn as nn
from torch import optim
from evaluate import *

import time
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

# -------------------------- Prepare Training Data -----------------------------

def indexesFromSentence(lang, sentence):
    """
    A helper function replacing tokens within a sentence by their index.
    :param lang: SCAN object with indexed data-structure
    :param sentence: A sentence
    :return: the sequence with token indices
    """
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    """
    A helper function computing in indexed sentences and appends EOS, returning this in form of a tensor.
    :param lang: SCAN object with indexed data-structure
    :param sentence: A sentence
    :return: a tensor representing a sentences by token indexes and an appended EOS token
    """
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(input_lang, output_lang, pair):
    """
    A helper function computing tensors from (command, action) pairs
    :param input_lang: SCAN object for the command sequence
    :param output_lang: SCAN object for the action sequence
    :param pair:
    :return: tensor pair existing of (command tensor, action tensor)
    """
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

# -------------------------- Keeping track of training time  -----------------------------
"""
    Helper functions for keeping track of training time
"""

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


# -------------------------- Training the Model -----------------------------
teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, iter, max_length, clip_norm_value=5.0):
    """
    Training the model on one training pair (input_tensor, target_tensor)
    :param input_tensor: input command tensor
    :param target_tensor: target action tensor
    :param encoder: encoder model that should be trained
    :param decoder: decoder model that should be trained
    :param encoder_optimizer: encoder_optimizer
    :param decoder_optimizer: decoder_optimizer
    :param criterion: loss function
    :param iter: number of current training iteration
    :param max_length: max_length
    :param clip_norm_value: clip_norm_value
    :return: loss at given training iteration of given training pair
    """

    # For LSTM also initialise cell state
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

    # Teacher forcing: feed 50% of iterations the target as the next input instead of own predictions
    use_teacher_forcing = True if iter < 100000 * 0.5 else False

    if use_teacher_forcing:
        for di in range(target_length):

            if 'Attn' in type(decoder).__name__:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        for di in range(target_length):

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


def trainIters(encoder, decoder, input_lang, output_lang, n_iters, max_length, learning_rate, clip_norm_value, pairs, test_pairs, plot_acc_file_path, plot_loss_file_path, result_file_path, experiment, run, prim, print_every=1000, is_experiment2=False):
    """
    Main function for training the encoder-decoder model
    :param encoder: encoder model that should be trained
    :param decoder: decoder model that should be trained
    :param input_lang: SCAN object for the command sequence
    :param output_lang: SCAN object for the action sequence
    :param n_iters: number of training iterations
    :param max_length: max length of command and action sequence
    :param learning_rate: learning_rate
    :param clip_norm_value: clip_norm_value
    :param pairs: training (command, action) pairs
    :param test_pairs: test (command, action) pairs
    :param plot_acc_file_path: file path for plotting accuracies
    :param plot_loss_file_path: file path for plotting loss
    :param result_file_path: file path for storing accuracy, loss results in table form
    :param print_every: specifying how often the model should be evaluated
    :param plot_every: specifying how often a result should be plotted
    :param is_experiment2: True, when Experiment 2, for special length evaluation, please refer for details to the paper
    """

    start = time.time()
    plot_losses_train = []
    plot_losses_test = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    input_max_length = max_length[0]
    output_max_length = max_length[1]

    # optimizer = AdamW
    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate)

    training_pairs = [tensorsFromPair(input_lang, output_lang, pair)
                      for pair in pairs]
    criterion = nn.NLLLoss()

    accs = []
    acc_lens = []
    iterations = []
    best_test_acc = 0

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

        best_test_loss = 1

        if (iter % print_every == 0):
            iterations += [iter]
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            encoder.eval()
            decoder.eval()

            acc, test_loss = evaluateTest(encoder, decoder, input_lang, output_lang, test_pairs, max_length, criterion)
            accs.append(acc)
            if is_experiment2:
                acc_len = evaluateTestLength(encoder, decoder, input_lang, output_lang, test_pairs, max_length)
                acc_lens.append(acc_len)

            plot_loss_avg = plot_loss_total / print_every
            plot_losses_train.append(plot_loss_avg)
            plot_loss_total = 0

            plot_losses_test.append(test_loss)

            # Saving the model with highest acc on test dataset (see individual reports for more details)
            if best_test_acc > acc:
                torch.save(encoder.state_dict(),
                           f'experiment_{experiment}/model/run_{run}_exp_{experiment}_state_dict_encoder_{prim}.pth')
                torch.save(decoder.state_dict(),
                           f'experiment_{experiment}/model/run_{run}_exp_{experiment}_state_dict_decoder_{prim}.pth')
                print(f'Model saved at iteration {iter} with accuracy of {acc} on test data')

    _dict = {'iteration': iterations, 'loss_train': plot_losses_train,
             'loss_test': plot_losses_test, 'acc_test': accs}
    pd.DataFrame(_dict).to_csv(result_file_path, index=False)

    if is_experiment2:
        showAccLenPlot(accs, acc_lens, iterations, plot_acc_file_path)
    else:
        showAccPlot(accs, iterations, plot_acc_file_path)
    showLossPlot(plot_losses_train, iterations, (plot_loss_file_path + str('_train.pdf')))
    showLossPlot(plot_losses_test, iterations, (plot_loss_file_path + str('_test.pdf')))


# -------------------------- PLOTTING RESULTS -----------------------------
"""
Helper functions for plotting results
"""

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def showAccPlot(points, iterations, plot_file_path):
    print (points)
    plt.plot(iterations, points)
    plt.xlabel('Step')
    plt.ylabel('Acc')
    plt.title('Evaluation Acc')
    plt.savefig(plot_file_path)
    plt.clf()

def showAccLenPlot(points1, points2, iterations, plot_file_path):
    plt.plot(iterations, points1, color='r', label="original")
    plt.plot(iterations, points2, color='b', label="with given lengths")
    plt.xlabel('Stesp')
    plt.ylabel('Acc')
    plt.title('Evaluation Acc')
    plt.legend()
    plt.savefig(plot_file_path)
    plt.clf()

def showLossPlot(points, iterations, plot_file_path):
    plt.plot(iterations, points)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.savefig(plot_file_path)
    plt.clf()

