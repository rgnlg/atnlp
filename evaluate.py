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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

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


def evaluate(encoder, decoder, input_lang, output_lang, pair, max_length, criterion):
    """
    Evaluating the given model (encoder, decoder) on a sentence, returning the decoded words
    :param encoder: the trained encoder object
    :param decoder: the trained decoder object
    :param input_lang: SCAN object for the command sequence
    :param output_lang: SCAN object for the action sequence
    :param pair: test pair (command, action)
    :param max_length: The computed max_length of commands and actions
    :return: The decoded words
    """

    # disable back-propagation
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, pair[0])
        input_length = input_tensor.size()[0]

        target_tensor = tensorFromSentence(output_lang, pair[1])
        target_length = target_tensor.size()[0]

        input_max_length = max_length[0]
        output_max_length = max_length[1]

        # initialising for LSTM cell an additional cell state
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

        loss = 0

        for di in range(output_max_length):

            # if attention-mechanism is applied, encoder_outputs are passed to the decoder
            if 'Attn' in type(decoder).__name__:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)

            topv, topi = decoder_output.data.topk(1)

            if di < target_length:
                loss += criterion(decoder_output, target_tensor[di])

            # print(topi)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, (loss.item() / target_length)


def evaluateTest(encoder, decoder, input_lang, output_lang, pairs, max_length, criterion):
    """
    Main function comparing decoded words with the gold labels and computes the acccuracy between them.
    :param encoder: the trained encoder object
    :param decoder: the trained decoder object
    :param input_lang: SCAN object for the command sequence
    :param output_lang: SCAN object for the action sequence
    :param pairs: (command, action) test pairs
    :param max_length: The computed max_length of commands and actions
    :return: Accuracy betweed decoded action tokens and gold action tokens
    """
    # settings models to evaluation mode (no backpropagation)
    encoder.eval()
    decoder.eval()

    truth = 0
    total = len(pairs)
    for pair in pairs:
        # print('>', pair[0])
        # print('=', pair[1])
        output_words, loss = evaluate(encoder, decoder, input_lang, output_lang, pair, max_length, criterion)
        output_sentence = ' '.join(output_words[:-1])
        # print('<', output_sentence)
        # print('')
        if output_sentence == pair[1]:
            truth += 1

    acc = truth/total
    print ("test_acc: {}".format(acc))
    print("test_loss: {}".format(loss))
    return acc, loss


def evaluateLength(encoder, decoder, input_lang, output_lang, sentence, max_length):
    """
    For Experiment 2: Evaluating with an oracle given the output sequence length during evaluation
    :param encoder: the trained encoder object
    :param decoder: the trained decoder object
    :param input_lang: SCAN object for the command sequence
    :param output_lang: SCAN object for the action sequence
    :param sentence: The sentence
    :param max_length: The computed max_length of commands and actions
    :return: The decoded words
    """
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

        for di in range(output_length):

            if 'Attn' in type(decoder).__name__:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)

            topv, topi = decoder_output.data.topk(2)
            if topi[0,0].item() == EOS_token:
                decoded_words.append(output_lang.index2word[topi[0,1].item()])
                decoder_input = topi.squeeze()[1].detach()
            else:
                decoded_words.append(output_lang.index2word[topi[0,0].item()])
                decoder_input = topi.squeeze()[0].detach()

        decoded_words.append('<EOS>')
        return decoded_words


def evaluateTestLength(encoder, decoder, input_lang, output_lang, pairs, max_length):
    """
    For Experiment 2: Evaluating with an oracle given the output sequence length during evaluation
    :param encoder: the trained encoder object
    :param decoder: the trained decoder object
    :param input_lang: SCAN object for the command sequence
    :param output_lang: SCAN object for the action sequence
    :param pairs: (command, action) test pairs
    :param max_length: The computed max_length of commands and actions
    :return: Accuracy betweed decoded action tokens and gold action tokens
    """
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


