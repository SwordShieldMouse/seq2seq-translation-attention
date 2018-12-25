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
from helper import *

data = Data()
input_size, output_size = data.get_vocab_lens()
input_lang, output_lang = data.get_names()

batch_size = 10
n_train = int(len(data) * 0.7)
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

# train
print("Training:")
for _ in range(epochs):
    for batch_x, batch_y in trainloader:
        # x, y are batches of sentences from input, output language
        loss = 0
        encoder_optim.zero_grad()
        attn_decoder_optim.zero_grad()
        for batch in range(len(batch_x)):
            encoder_hidden = encoder.reset_hidden()

            # convert the sentence pair to tensors
            input_tensor, output_tensor = tensors_from_pair(data.input_vocab, data.output_vocab, (batch_x[batch], batch_y[batch]))

            input_len = input_tensor.size(0)
            output_len = output_tensor.size(0)

            encoder_outputs = torch.zeros([MAX_LENGTH, hidden_size])

            # run the encoder
            for ix in range(input_len):
                encoder_output, encoder_hidden = encoder(input_tensor[ix], encoder_hidden)
                encoder_outputs[ix] = encoder_output[0][0]

            # run the decoder
            # feed the SOS token as input
            decoder_input = torch.tensor([[SOS_token]])
            # use the last hidden state of the encoder as the 1st hidden state of the decoder
            decoder_hidden = encoder_hidden

            if np.random.random() < teacher_forcing_ratio:
                use_teacher_forcing = True
            else:
                use_teacher_forcing = False

            if use_teacher_forcing is True:
                for ix in range(output_len):
                    decoder_output, decoder_hidden, attn_weights = attn_decoder(decoder_input, decoder_hidden, encoder_outputs)
                    loss += criterion(decoder_output, output_tensor[ix])
                    decoder_input = output_tensor[ix] # teacher forcing rather than actual output
            else:
                for ix in range(output_len):
                    decoder_output, decoder_hidden, attn_weights = attn_decoder(decoder_input, decoder_hidden, encoder_outputs)
                    topv, topi = decoder_output.topk(1)

                    # detach from graph
                    # can just be index since we pass it into an embedding
                    decoder_input = topi.squeeze().detach() # no teacher forcing, so use the actual output

                    loss += criterion(decoder_output, output_tensor[ix])
                    if decoder_input.item() == EOS_token:
                        break

        loss.backward()

        encoder_optim.step()
        attn_decoder_optim.step()

print("Training complete")

# evaluate
for x, y in testloader:
    pred = evaluate(encoder, attn_decoder, data, x, y)
    print("Original is {}".format(x))
    print("Translation is {}".format(' '.join(pred)))
    break
