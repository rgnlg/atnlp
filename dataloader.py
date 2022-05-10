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
import random
import torch


SOS_token = 0
EOS_token = 1

class SCAN:
    """
    SCAN class maps tokens to indexes from a given dataset.
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {} #word-to-index dictionary
        self.word2count = {} #count of each word
        self.index2word = {0: "SOS", 1: "EOS"}  # index-to-word dictionary
        self.n_words = 2  #count SOS and EOS
        self.max_length = 0 #save the max length

    def addSentence(self, sentence):
        """
        A SCAN helper function splitting sentences into a sequence of words and store the set of words in an indexed data-structure as well as the max length of all sentences.
        :param sentence: A sentence.
        """
        for word in sentence.split(' '):
            self.addWord(word)
        if self.max_length < len(sentence.split(' ')):
            self.max_length = len(sentence.split(' '))

    def addWord(self, word):
        """
        A simple SCAN helper function adding a word to the SCAN data-structures and word counting.
        :param word: A single token.
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_txt(data_path):
    """
    Reading the txt data and filerting the (command, action) pair by "IN:" and "OUT:" indicators.
    :param data_path: path to file
    :return: (command, action) pairs
    """
    pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            pair = line.replace("IN:", "").split("OUT:")
            pair = [string.strip() for string in pair]
            pairs.append(pair)
    return pairs


def oversampling(pairs):
    """
    Oversamples the given training pairs to 100,000 samples by using random sampling with replacement.
    :param pairs: training pairs (command, action)
    :return: Final pairs existing of the original training pairs extended by the additional sampling pairs.
    """
    sampler = torch.utils.data.sampler.RandomSampler(range(len(pairs)), replacement=True, num_samples=100000-len(pairs))

    random_pairs = []
    for idx in sampler:
        random_pairs.append(pairs[idx])

    final_pairs = pairs + random_pairs
    random.shuffle(final_pairs)

    return final_pairs


def prepare_data(train_file_path, test_file_path, do_oversampling=False):
    """
    A preprocessing function loading train and test dataset and loading all commands and action sequences and training, test pairs.
    :param train_file_path: the path to the train dataset
    :param test_file_path: the path to the test dataset
    :param do_oversampling: if True, applies oversampling to 100,000 training pairs
    :return: SCAN object of commands, SCAN object of actions, training pairs, testing pairs
    """

    train_pairs = read_txt(train_file_path)
    test_pairs = read_txt(test_file_path)

    command_set = SCAN("command")
    action_set = SCAN("action")

    for pair in train_pairs:
        command_set.addSentence(pair[0])
        action_set.addSentence(pair[1])

    for pair in test_pairs:
        command_set.addSentence(pair[0])
        action_set.addSentence(pair[1])

    print("Counted words:")
    print("{}: {}".format(command_set.name, command_set.word2index.keys()))
    print("{}: {}".format(action_set.name, action_set.word2index.keys()))

    if do_oversampling:
        train_pairs = oversampling(train_pairs)

    return command_set, action_set, train_pairs, test_pairs


if __name__ == '__main__':
    # for testing
    train_file_path = "../SCAN/simple_split/tasks_train_simple.txt"
    test_file_path = "../SCAN/simple_split/tasks_test_simple.txt"
    command_set, action_set, train_pairs, test_pairs = prepare_data(train_file_path, test_file_path,
                                                                    do_oversampling=True)
    print("The length of training set: {}".format(len(train_pairs)))
    print("The length of test set: {}".format(len(test_pairs)))
    # print(random.choice(train_pairs))