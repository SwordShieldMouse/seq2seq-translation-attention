import torch
import torch.nn as nn
import re
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import unicodedata

SOS_token = 0 # start of string
EOS_token = 1 # end of string

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def reset_hidden(self):
        return torch.zeros(1, self.hidden_size)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # can't we somehow share embeddings with the encoder?
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)

    # takes hidden as an argument which allows us to use teacher forcing
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def reset_hidden(self):
        return torch.zeros(1, self.hidden_size)

class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p = 0.1, max_length = MAX_LENGTH):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_length) # attention over the outputs of the encoder (the sentence)

        # combine the attention-weighted encoder outputs with the embedding
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim = 1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim = 1)

        return output, hidden, attn_weights

    def reset_hidden(self):
        return torch.zeros(1, self.hidden_size)


# helps use keep track of the vocabulary
class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2ix = {}
        self.word2count = {}
        self.ix2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 # the SOS and EOS tokens

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2ix.keys():
            self.word2ix[word] = self.n_words
            self.word2count[word] = 1
            self.ix2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def ixs_from_sentence(vocab, sentence):
    return [vocab.word2ix[word] for word in sentence.split(' ')]

def tensor_from_sentence(vocab, sentence):
    # returns column of indices of words in sentence
    ixs = ixs_from_sentence(vocab, sentence)
    ixs.append(EOS_token)
    return torch.tensor(ixs).view(-1, 1)

def tensors_from_pair(input_vocab, output_vocab, pair):
    input_tensor = tensor_from_sentence(input_vocab, pair[0])
    target_tensor = tensor_from_sentence(output_vocab, pair[1])
    return (input_tensor, target_tensor)

class Data(Dataset):
    def __init__(self):
        self.input_vocab, self.output_vocab, self.pairs = read_languages('eng', 'fra', True)
        print("read in {0} language pairs".format(len(self.pairs)))
        self.pairs = filter_pairs(self.pairs)
        print("filtered to {} language pairs".format(len(self.pairs)))
        for pair in self.pairs:
            self.input_vocab.add_sentence(pair[0])
            self.output_vocab.add_sentence(pair[1])
        print("word count:")
        print(self.input_vocab.name, self.input_vocab.n_words)
        print(self.output_vocab.name, self.output_vocab.n_words)

    def __len__(self):
        # returns length of the pairs dataset
        return len(self.pairs)

    def get_vocab_lens(self):
        # returns lengths of input, output vocabs
        # use ix2word to account for SOS and EOS tokens
        return self.input_vocab.n_words, self.output_vocab.n_words

    def get_names(self):
        return self.input_vocab.name, self.output_vocab.name

    def __getitem__(self, ix):
        return self.pairs[ix]


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# lower-case, trim, remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'[.!?]', r'  ', s) # leave a double space to separate sentences
    s = re.sub(r'[^a-zA-Z.!?]', r' ', s)
    return s

def read_languages(lang1, lang2, reverse = False):
    print("Reading the lines. ")

    lines = []

    with open("data/{0}-{1}.txt".format(lang1, lang2), encoding = 'utf-8') as file:
        lines = [line.strip() for line in file]

    # split the lines into language pairs and normalize
    pairs = [[normalize_string(s) for s in line.split('\t')] for line in lines]

    if reverse is True:
        pairs = [list(reversed(p)) for p in pairs]
        input_vocab = Vocab(lang2)
        output_vocab = Vocab(lang1)
    else:
        input_vocab = Vocab(lang1)
        output_vocab = Vocab(lang2)

    return input_vocab, output_vocab, pairs

def filter_pair(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(" ")) < MAX_LENGTH and pair[1].startswith(eng_prefixes)

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def evaluate(encoder, decoder, data, x, y, max_length = MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(data.input_vocab, x)
        input_length = input_tensor.size(0)

        encoder_outputs = torch.zeros((MAX_LENGTH, encoder.hidden_size))
        encoder_hidden = encoder.reset_hidden()

        for ix in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ix], encoder_hidden)
            encoder_outputs[ix] = encoder_output[0]

        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden
        decoded_words = []

        for ix in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(data.output_lang.ix2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words
