import glob
import os
import string
import unicodedata
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

# holds all the main classes
import helper

data = Data()
input_size, output_size = data.get_vocab_lens()
input_lang, output_lang = data.get_names()

batch_size = 10
n_train = int(len(data) * 0.8)
n_test = len(data) - n_train

epochs = 3
hidden_size = 128
learning_rate = 1e-4


train_data, test_data = torch.utils.data.random_split(data, [n_train, n_test])

trainloader = DataLoader(train_data, shuffle = True, batch_size = batch_size)
testloader = DataLoader(test_data, shuffle = True)

teacher_forcing_ratio = 0.5

encoder = EncoderRNN(input_size, hidden_size)
decoder = DecoderRNN(hidden_size, output_size)
attn_decoder = AttentionDecoderRNN(hidden_size, output_size)

criterion = nn.NLLLoss()
encoder_optim = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
decoder_optim = torch.optim.Adam(decoder.parameters(), lr = learning_rate)
attn_decoder_optim = torch.optim.Adam(attn_decoder.parameters(), lr = learning_rate)

for _ in range(epochs):
    for x, y in trainloader:
        # x, y are batches of sentences from input, output language
        encoder_hidden = encoder.reset_hidden()

        encoder_optim.zero_grad()
        attn_decoder_optim.zero_grad()

        
