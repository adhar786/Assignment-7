#%%
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
#to get the data, go to https://www.gutenberg.org/. Then go to search and browse, then go to animals. I chose a book randomly and downloaded it to a txt file
#Let me know if you need help with that, it wasn't super apparent how to download it

with open('/content/BGLLM_1.txt', "r", encoding="utf-8") as file:
        text = file.read()
data=html.unescape(text)
sentences=sent_tokenize(text)

#%%
def cleanUp(data):
  #tokenize using WhitespaceTokenizer, which includes some punctuation to keep contractions as single word
  data=data.translate(str.maketrans('', '', string.punctuation))
  words=nltk.WhitespaceTokenizer().tokenize(data)
  #remove stop words, punctuation, make lowercase
  cleaned= [w.lower() for w in words if not w.isnumeric()]
  return cleaned

#clean up all reviews
tokenized=[cleanUp(i) for i in sentences]
#convert to int
#the first sentence is always the same and has some weird token I can't get rid of
print(len(tokenized))
#covert to words to integers
word_model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1)
#use Word2Vec's method wv to get a KeyedVectors object, which stores the unique vector related to each word in the reviews
embeddings=word_model.wv
print(f'vocab size is {len(embeddings)}')

tokenized=tokenized[1:]
tokenized=[np.array(i) for i in tokenized]
int_sequences =[[embeddings[word] for word in sentence if word in embeddings] for sentence in tokenized]
print(np.shape(int_sequences[0]))
# Pad sequences
max_length = 10
padded_sentences = np.zeros((10,100,len(int_sequences)))
print(np.shape(padded_sentences[:,:,0]))
for counter,sentence in enumerate(int_sequences):
    num_padding = max_length - np.shape(sentence)[0]
    if num_padding > 0:
        if np.array(sentence).ndim == 1:
          continue
        padded_sentence = np.pad(sentence, ((0,num_padding),(0,0)), mode='constant', constant_values=0)

    else:
        padded_sentence = np.array(sentence)[:max_length,:]


    padded_sentences[:,:,counter]=padded_sentence

# Convert to array
print(np.shape(padded_sentences))
padded_sentences = np.array(padded_sentences)

#%%
encoder_input = padded_sentences[:-1,: :]
decoder_input=padded_sentences[:-1,: :]
decoder_target=padded_sentences[-1,:,:]
print(np.shape(decoder_target))
#%%
def create_mask(batch_size, tokens, ndims):
  import torch
  mask=torch.full((10,100),float('-inf'))
  return torch.triu(mask,1)

print(create_mask(1,10,100))