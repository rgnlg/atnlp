"""
Authors: Duo Yang (vdb927)
Olga Iarygina (hwk263)
Philine Zeinert (vdp117)

Re-Implementation of:
Lake, B. M. and Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. International Conference on Machine Learning (ICML)
(ATNLP course, Copenhagen University, 2021/22)

Date: 28/01/2022
"""

from __future__ import unicode_literals, print_function, division
import os
import json
import argparse
import random
from pathlib import Path
from train import trainIters
from dataloader import prepare_data
from model import *
from evaluate import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configurations_path', type=str, required=True)
    parser.add_argument('--run', type=str, required=True)
    args = parser.parse_args()

    with open(args.configurations_path, 'r') as j:
        conf_ = json.loads(j.read())
    # loading experiment configurations
    experiment = conf_['experiment']
    train_file_path = conf_['train_file_path']
    test_file_path = conf_['test_file_path']
    is_train = conf_['is_train']
    is_over_all_best = conf_['is_over_all_best']
    prim = conf_['prim']

    # creating directory tree for saving results
    if not os.path.exists(Path('experiment_' + str(experiment))):
        Path('experiment_' + str(experiment)).mkdir(parents=True, exist_ok=True)
    mode = 0o666
    parent_dir = 'experiment_' + str(experiment)
    subdirs = ['model','results']
    for subdir in subdirs:
        subdir_path = Path(parent_dir, subdir)
        if not os.path.exists(subdir_path):
            subdir_path.mkdir(parents=True, exist_ok=True)
            print("Directory '% s' created" % subdir)
    plot_acc_file_path = f'experiment_{experiment}/results/run_{args.run}_exp_{experiment}_plot_acc_{prim}.pdf'
    plot_loss_file_path = f'experiment_{experiment}/results/run_{args.run}_exp_{experiment}_plot_loss_{prim}'
    result_file_path = f'experiment_{experiment}/results/run_{args.run}_exp_{experiment}_results_{prim}.csv'

    # loading data
    command_set, action_set, train_pairs, test_pairs = prepare_data(train_file_path, test_file_path, do_oversampling=True)
    print("The length of training set: {}".format(len(train_pairs)))
    print("The length of test set: {}".format(len(test_pairs)))
    print(random.choice(train_pairs))

    # setting training variables according to the paper
    iteration = 100000
    clip_norm_value = 5.0
    learning_rate = 0.001
    criterion = nn.NLLLoss()

    # saving max. lengths of source and target sequences and vocab size by number of words
    input_max_length = command_set.max_length + 1
    output_max_length = action_set.max_length + 2
    max_length = [input_max_length, output_max_length]

    encoder_vocab_size = command_set.n_words
    decoder_vocab_size = action_set.n_words


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading the model
    if is_over_all_best:

        dropout_rnn = 0.5
        dropout_embedded = 0.5
        hidden_size = 200
        num_layers = 2

        encoder = EncoderLSTM(encoder_vocab_size, hidden_size, num_layers, dropout_rnn, dropout_embedded).to(device)
        decoder = DecoderLSTM(hidden_size, decoder_vocab_size, num_layers, dropout_rnn, dropout_embedded).to(device)

    # best model experiment 1
    elif experiment=='1':
        dropout_p = 0
        dropout_embedded = 0
        hidden_size = 200
        num_layers = 2

        encoder = EncoderLSTM(encoder_vocab_size, hidden_size, num_layers, dropout_p, dropout_embedded).to(device)
        decoder = DecoderLSTM(hidden_size, decoder_vocab_size, num_layers, dropout_p, dropout_embedded).to(device)


    # best model experiment 2
    elif experiment=='2':

        dropout_rnn = 0.5
        dropout_embedded = 0.5
        hidden_size = 50
        num_layers = 1

        encoder = EncoderGRU(encoder_vocab_size, hidden_size, num_layers, dropout_rnn, dropout_embedded).to(device)
        decoder = AttnDecoder_layer_GRU(hidden_size, decoder_vocab_size, num_layers, input_max_length, dropout_rnn,
                                        dropout_embedded).to(device)

    # best models experiment 3
    elif experiment == '3':
        if prim == "jump":

            dropout_rnn = 0.1
            dropout_embedded = 0.1
            hidden_size = 100
            num_layers = 1

            encoder = EncoderLSTM(encoder_vocab_size, hidden_size, num_layers, dropout_rnn, dropout_embedded).to(device)
            decoder = AttnDecoder_layer_LSTM(hidden_size, decoder_vocab_size, num_layers, input_max_length, dropout_rnn, dropout_embedded).to(
            device)

        elif prim == "turn_left":
            dropout_rnn = 0.1
            dropout_embedded = 0.1
            hidden_size = 100
            num_layers = 1

            encoder = EncoderGRU(encoder_vocab_size, hidden_size, num_layers, dropout_rnn, dropout_embedded).to(device)
            decoder = AttnDecoder_layer_GRU(hidden_size, decoder_vocab_size, num_layers, input_max_length, dropout_rnn,
                                            dropout_embedded).to(device)

        else:
            print('Please specify in Experiment 3 the "prim": Either "jump" or "turn_left".')


    if is_train:

        if experiment=='2':
            is_experiment2 = True

        trainIters(encoder, decoder, command_set, action_set, iteration, max_length, learning_rate, clip_norm_value,train_pairs, test_pairs, plot_acc_file_path, plot_loss_file_path, result_file_path, experiment, args.run, prim, print_every=5000, is_experiment2=False)

        evaluateTest(encoder, decoder, command_set, action_set, test_pairs, max_length, criterion)
        if experiment == '2':
            evaluateTestLength(encoder, decoder, command_set, action_set, test_pairs, max_length)

    else:

        encoder.load_state_dict(torch.load(f'experiment_{experiment}/model/run_{args.run}_exp_{experiment}_state_dict_encoder_{prim}.pth'))
        decoder.load_state_dict(torch.load(f'experiment_{experiment}/model/run_{args.run}_exp_{experiment}_state_dict_decoder_{prim}.pth'))

        evaluateTest(encoder, decoder, command_set, action_set, test_pairs, max_length, criterion)
        if experiment == '2':
            evaluateTestLength(encoder, decoder, command_set, action_set, test_pairs, max_length)