#from "Evaluation" - https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#loading-data-files

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

# Same as training, but there are no targets - feed the decoder's predicitons back to itself for each step
# every time it predicts a word, we add it to the output stirng,
# and if it predicts the EOS token we stop there
# also store decoder's attention outputs for idsplay later

SOS_token = 0
EOS_token = 1

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

def evaluate(encoder, decoder, input_lang, output_lang, sentence, max_length):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        # encoder_hidden = encoder.initHidden()

        input_max_length = max_length[0]
        output_max_length = max_length[1]

        if 'LSTM' in type(encoder).__name__:
            encoder_h = encoder.initHidden()
            encoder_c = encoder.initCell()
            encoder_hidden = [encoder_h, encoder_c]
        else:
            encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(input_max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        # decoder_attentions = torch.zeros(output_max_length, input_max_length)

        for di in range(output_max_length):

            if 'Attn' in type(decoder).__name__:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)

            #decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            #decoder_attentions[di] = decoder_attention.data

            topv, topi = decoder_output.data.topk(1)
            # print(topi)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        # return decoded_words, decoder_attentions[:di + 1]
        return decoded_words

def evaluateTest(encoder, decoder, input_lang, output_lang, pairs, max_length):

    encoder.eval()
    decoder.eval()

    truth = 0
    total = len(pairs)
    for pair in pairs:
        # print('>', pair[0])
        # print('=', pair[1])
        # output_words, attentions = evaluate(encoder, decoder, input_lang, output_lang, pair[0], max_length)
        output_words = evaluate(encoder, decoder, input_lang, output_lang, pair[0], max_length)
        output_sentence = ' '.join(output_words[:-1])
        # print('<', output_sentence)
        # print('')
        if output_sentence == pair[1]:
            truth += 1
    acc = truth/total
    print ("acc: {}".format(acc))

    return acc

def evaluateLength(encoder, decoder, input_lang, output_lang, sentence, max_length):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence[0])
        output_length = len(sentence[1].split())
        input_length = input_tensor.size()[0]
        # encoder_hidden = encoder.initHidden()

        input_max_length = max_length[0]
        output_max_length = max_length[1]

        if 'LSTM' in type(encoder).__name__:
            encoder_h = encoder.initHidden()
            encoder_c = encoder.initCell()
            encoder_hidden = [encoder_h, encoder_c]
        else:
            encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(input_max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        # decoder_attentions = torch.zeros(output_max_length, input_max_length)

        for di in range(output_length):

            if 'Attn' in type(decoder).__name__:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)

            #decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            #decoder_attentions[di] = decoder_attention.data

            topv, topi = decoder_output.data.topk(2)
            if topi[0,0].item() == EOS_token:
                decoded_words.append(output_lang.index2word[topi[0,1].item()])
                decoder_input = topi.squeeze()[1].detach()
            else:
                decoded_words.append(output_lang.index2word[topi[0,0].item()])
                decoder_input = topi.squeeze()[0].detach()

        decoded_words.append('<EOS>')
        # print (decoded_words)

        # return decoded_words, decoder_attentions[:di + 1]
        return decoded_words

def evaluateTestLength(encoder, decoder, input_lang, output_lang, pairs, max_length):

    encoder.eval()
    decoder.eval()

    truth = 0
    total = len(pairs)
    for pair in pairs:
        # print('>', pair[0])
        # print('=', pair[1])
        # output_words, attentions = evaluate(encoder, decoder, input_lang, output_lang, pair[0], max_length)
        output_words = evaluateLength(encoder, decoder, input_lang, output_lang, pair, max_length)
        output_sentence = ' '.join(output_words[:-1])
        # print('<', output_sentence)
        # print('')
        if output_sentence == pair[1]:
            truth += 1

    acc = truth/total
    print ("acc (given length): {}".format(acc))

    return acc


