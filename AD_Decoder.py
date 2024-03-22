# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import nltk
import string
import numpy as np
from gensim.models import Word2Vec
import nltk
import numpy as np
import pandas as pd
import keras
import tensorflow
import string
import os
import html
from gensim.models import Word2Vec
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from gensim.test.utils import common_texts
import regex as re

# %%
# to get the data, go to https://www.gutenberg.org/. Then go to search and browse, then go to animals. I chose a book randomly and downloaded it to a txt file
# Let me know if you need help with that, it wasn't super apparent how to download it

with open('BGLLM_1.txt', "r", encoding="utf-8") as file:
    text = file.read()
data = html.unescape(text)
sentences = sent_tokenize(text)


# %%
def cleanUp(data):
    # tokenize using WhitespaceTokenizer, which includes some punctuation to keep contractions as single word
    data = data.translate(str.maketrans('', '', string.punctuation))
    words = nltk.WhitespaceTokenizer().tokenize(data)
    # remove stop words, punctuation, make lowercase
    cleaned = [w.lower() for w in words if not w.isnumeric()]
    return cleaned


def vocab_dictionary(list_of_list_of_words):
    # Flatten the list of lists into a single list of tokens
    all_tokens = [token for sublist in list_of_list_of_words for token in sublist]

    # Use a set to find unique tokens, then sort them (optional, for consistency)
    unique_tokens = sorted(set(all_tokens))

    # Create a dictionary mapping each unique token to a unique integer
    out_vocab = {token: idx for idx, token in enumerate(unique_tokens)}
    return out_vocab


def get_keys_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    return keys


def truncate_and_pad(list_of_lists_of_tokens, desired_length, pad_symbol):
    output = []
    for sentence in list_of_lists_of_tokens:
        if len(sentence) >= desired_length:
            sentence = sentence[:desired_length]
        else:
            for i in range(desired_length-len(sentence)):
                sentence.append(pad_symbol)
        # manually add sos to each sentence
        sentence = [-2] + sentence
        output.append(sentence)
    return output


# clean up all reviews
tokenized = [cleanUp(i) for i in sentences]
vocab = vocab_dictionary(tokenized)
# add sos, eos, pad, to vocab
vocab['<pad>'] = -1
vocab['<sos>'] = -2
vocab['<eos>'] = -3
# convert to int
tokenized_int = list([vocab.get(word) for word in sentence] for sentence in tokenized)
# for the uniform_int_tokens, we manually add sos and eos to start and end of each sentence
# so the length of each sentence becomes 12
uniform_int_tokens = truncate_and_pad(tokenized_int, 10, -1)
print(f'the length of each sentence is {len(uniform_int_tokens[0])}')
print(f'the number of sentences is {len(uniform_int_tokens)}')
# the first sentence is always the same and has some weird token I can't get rid of
print(len(tokenized))
# covert to words to integers
word_model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1)
# use Word2Vec's method wv to get a KeyedVectors object, which stores the unique vector related to each word in
# the reviews
embeddings = word_model.wv
print(f'vocab size is {len(embeddings)}')

tokenized = tokenized[1:]
tokenized = [np.array(i) for i in tokenized]
int_sequences = [[embeddings[word] for word in sentence if word in embeddings] for sentence in tokenized]
print(np.shape(int_sequences[0]))
# Pad sequences
max_length = 10
padded_sentences = np.zeros((10, 100, len(int_sequences)))
print(np.shape(padded_sentences[:, :, 0]))
for counter, sentence in enumerate(int_sequences):
    num_padding = max_length - np.shape(sentence)[0]
    if num_padding > 0:
        if np.array(sentence).ndim == 1:
            continue
        padded_sentence = np.pad(sentence, ((0, num_padding), (0, 0)), mode='constant', constant_values=0)

    else:
        padded_sentence = np.array(sentence)[:max_length, :]

    padded_sentences[:, :, counter] = padded_sentence

# Convert to array
print(np.shape(padded_sentences))
padded_sentences = np.array(padded_sentences)

# %%
encoder_input = padded_sentences[:-1, ::]
decoder_input = np.array(uniform_int_tokens)[:, :-1]
decoder_target = np.array(uniform_int_tokens)[:, 1:]
print(f'the shape of the decoder target is {np.shape(decoder_target)}')
print(f'the shape of the encoder input is {np.shape(encoder_input)}')
print(f'the shape of the decoder input is {np.shape(decoder_input)}')
np.save('encoder_input', encoder_input)
np.save('decoder_input', decoder_input)
np.save('decoder_target', decoder_target)


# %%
def create_mask(batch_size, tokens, ndims):
    import torch
    mask = torch.full((10, 100), float('-inf'))
    return torch.triu(mask, 1)


# print(create_mask(1, 10, 100))
