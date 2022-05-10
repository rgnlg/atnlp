#from "Loading data files" - https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#loading-data-files:
from __future__ import unicode_literals, print_function, division
import random
import numpy as np
import torch


SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {} #word-to-index dictionary
        self.word2count = {} #count of each word (used to replace rare words later)
        self.index2word = {0: "SOS", 1: "EOS"} #index-to-word dictionary
        self.n_words = 2  #Count SOS and EOS
        self.max_length = 0 #save the max length

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
        if self.max_length < len(sentence.split(' ')):
            self.max_length = len(sentence.split(' '))

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_txt(data_path):
    pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            pair = line.replace("IN:", "").split("OUT:")
            pair = [ string.strip() for string in pair]
            pairs.append(pair)
    return pairs


def oversampling(pairs):

    sampler = torch.utils.data.sampler.RandomSampler(range(len(pairs)), replacement=True, num_samples=100000-len(pairs))

    random_pairs = []
    for idx in sampler:
        random_pairs.append(pairs[idx])

    final_pairs = pairs + random_pairs
    random.shuffle(final_pairs)

    return final_pairs


def prepare_data(train_file_path, test_file_path, do_oversampling=False):

    train_pairs = read_txt(train_file_path)
    test_pairs = read_txt(test_file_path)

    command_set = Lang("command")
    action_set = Lang("action")

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


def main():

    train_file_path = "../SCAN/simple_split/tasks_train_simple.txt"
    test_file_path = "../SCAN/simple_split/tasks_test_simple.txt"
    command_set, action_set, train_pairs, test_pairs = prepare_data(train_file_path, test_file_path, do_oversampling=True)
    print("The length of training set: {}".format(len(train_pairs)))
    print("The length of test set: {}".format(len(test_pairs)))
    # print(random.choice(train_pairs))


if __name__ == '__main__':
    main()