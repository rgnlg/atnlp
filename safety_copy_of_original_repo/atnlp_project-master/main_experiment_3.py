from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from train import trainIters
from dataloader import prepare_data
from model import *
from evaluate import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--run', type=str, required=True)
    parser.add_argument('--train_file_path', type=str, required=True)
    parser.add_argument('--test_file_path', type=str, required=True)
    #parser.add_argument('--plot_acc_file_path', type=str, required=True)
    #parser.add_argument('--plot_loss_file_path', type=str, required=True)
    #parser.add_argument('--result_file_path', type=str, required=True)
    parser.add_argument('--is_train', type=bool, default=False)
    parser.add_argument('--is_over_all_best', type=bool, default=False)
    parser.add_argument('--add_prim', type=str, choices=["jump", "turn_left"], required=True,
                        help='Choose between ["jump", "turn_left"]')
    args = parser.parse_args()


    if not os.path.exists(Path('experiment_' + str(args.experiment))):
        Path('experiment_' + str(args.experiment)).mkdir(parents=True, exist_ok=True)
    mode = 0o666
    parent_dir = 'experiment_' + str(args.experiment)

    subdirs = ['model','results']
    for subdir in subdirs:
        subdir_path = Path(parent_dir, subdir)
        if not os.path.exists(subdir_path):
            subdir_path.mkdir(parents=True, exist_ok=True)
            print("Directory '% s' created" % subdir)
    plot_acc_file_path = f'experiment_{args.experiment}/results/plot_acc_{args.add_prim}_{args.run}.pdf'
    plot_loss_file_path = f'experiment_{args.experiment}/results/plot_loss_{args.add_prim}_{args.run}.pdf'
    result_file_path = f'experiment_{args.experiment}/results/results_{args.add_prim}_{args.run}.csv'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    command_set, action_set, train_pairs, test_pairs = prepare_data(args.train_file_path, args.test_file_path, do_oversampling=True)
    print("The length of training set: {}".format(len(train_pairs)))
    print("The length of test set: {}".format(len(test_pairs)))
    print(random.choice(train_pairs))

    iteration = 100000
    clip_norm_value = 5.0
    learning_rate = 0.001

    input_max_length = command_set.max_length + 1
    output_max_length = action_set.max_length + 2
    max_length = [input_max_length, output_max_length]

    encoder_vocab_size = command_set.n_words
    decoder_vocab_size = action_set.n_words

    if args.is_over_all_best:

        dropout_rnn = 0.5
        dropout_embedded = 0.5
        hidden_size = 200
        num_layers = 2

        encoder = EncoderLSTM(encoder_vocab_size, hidden_size, num_layers, dropout_rnn, dropout_embedded).to(device)
        decoder = DecoderLSTM(hidden_size, decoder_vocab_size, num_layers, dropout_rnn, dropout_embedded).to(device)

    elif (~args.is_over_all_best) & (args.add_prim == "jump"):

        dropout_rnn = 0.1
        dropout_embedded = 0.1
        hidden_size = 100
        num_layers = 1

        encoder = EncoderLSTM(encoder_vocab_size, hidden_size, num_layers, dropout_rnn, dropout_embedded).to(device)
        decoder = AttnDecoder_layer_LSTM(hidden_size, decoder_vocab_size, num_layers, input_max_length, dropout_rnn, dropout_embedded).to(
            device)

    elif (~args.is_over_all_best) & (args.add_prim == "turn_left"):
        dropout_rnn = 0.1
        dropout_embedded = 0.1
        hidden_size = 100
        num_layers = 1

        encoder = EncoderGRU(encoder_vocab_size, hidden_size, num_layers, dropout_rnn, dropout_embedded).to(device)
        decoder = AttnDecoder_layer_GRU(hidden_size, decoder_vocab_size, num_layers, dropout_rnn, input_max_length, dropout_embedded).to(
            device)


    if args.is_train:

        trainIters(encoder, decoder, command_set, action_set, iteration, max_length, learning_rate, clip_norm_value,train_pairs, test_pairs, plot_acc_file_path, plot_loss_file_path, result_file_path, print_every=5000, plot_every=5000, is_experiment2=True)

        torch.save(encoder.state_dict(),f'experiment_{args.experiment}/model/state_dict_encoder_{args.add_prim}_{args.run}.pth')
        torch.save(decoder.state_dict(), f'experiment_{args.experiment}/model/state_dict_decoder_{args.add_prim}_{args.run}.pth')

        evaluateTest(encoder, decoder, command_set, action_set, test_pairs, max_length)
        evaluateTestLength(encoder, decoder, command_set, action_set, test_pairs, max_length)


    else:

        encoder.load_state_dict(torch.load(f'experiment_{args.experiment}/model/state_dict_encoder_{args.add_prim}_{args.run}.pth'))
        decoder.load_state_dict(torch.load(f'experiment_{args.experiment}/model/state_dict_decoder_{args.add_prim}_{args.run}.pth'))

        evaluateTest(encoder, decoder, command_set, action_set, test_pairs, max_length)
        evaluateTestLength(encoder, decoder, command_set, action_set, test_pairs, max_length)


if __name__ == '__main__':
    main()
