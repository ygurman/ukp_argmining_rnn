## model package for the 2-part tagger - (biLSTM AC segmentor-classifier & encoder-decoder relation classifier)
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from src.relations.preprocess import ArgComp


class thinRelationClassifier(nn.Module):
    """
    a lean linear model for relation classification consisting of ACs display using dotwise mean
    followed by matrix multiplication in order to create a 100x100 representation of every two ACs
    then through
    """

class biLSTMRelationClassifier(nn.Module):
    def __init__(self, d_word_embd, d_pos_embd,
                 d_h1, n_lstm_layers, word_voc_size,
                 pos_voc_size, ac_tagset_size, batch_size,
                 device, pretraind_embd_layer_path=None,
                 d_tag_embd,):
        super().__init__()

        ## part 1 - using pretrained segmentor classifier up until LSTM layer representation
        self.device = device
        self.h1dimension = d_h1
        self.batch_size = batch_size
        # embedding layers (words and POS tags)
        self.embd_word_layer = pickle.load(open(pretraind_embd_layer_path, 'rb')) if pretraind_embd_layer_path \
            else nn.Embedding(num_embeddings=word_voc_size, embedding_dim=d_word_embd)
        self.embd_pos_layer = nn.Embedding(num_embeddings=pos_voc_size, embedding_dim=d_pos_embd)
        # sequence (bilstm) layer - recive previous concatenated embeddings and pass to linear classification (for tagging)
        self.lstm1 = nn.LSTM(input_size=d_word_embd + d_pos_embd,  # input as concatenated word|pos embeddings
                             hidden_size=d_h1,
                             # hidden state parameter (for token level AC class) to pass to the final linear layer
                             num_layers=n_lstm_layers,
                             # number of LSTM layers (set to 1 but passed as parameter for reusability)
                             bidirectional=True)  # sets as bi-drectional LSTM

        # part 2 - relation classification extraction
        # embedding the constructed features in order to be able to learn their contribution to the prediction
        self.embd_ac_tag_layer = nn.Embedding(num_embeddings=ac_tagset_size,embedding_dim=d_tag_embd)
        self.embd_a_is_before = nn.Embedding(num_embeddings=2,embedding_dim=5) #boolen
        self.embd_in_same_par = nn.Embedding(num_embeddings=2,embedding_dim=5) # boolean
        self.embd_distance = nn.Embedding(num_embeddings=30,embedding_dim=20) # bounded by the maximal id



        # initialize first hidden states
        self.hidden1 = self.init_hidden(self.h1dimension)

    def init_hidden(self, hidden_dim):
        """
        initialize hidden states of the biLSTM
        :param hidden_dim:
        :return:
        """
        # initialize hidden states with zeroes (2 because bidirectional lstm)
        h0 = autograd.Variable(torch.zeros(1 * 2, self.batch_size, hidden_dim)).to(self.device)
        c0 = autograd.Variable(torch.zeros(1 * 2, self.batch_size, hidden_dim)).to(self.device)
        return h0, c0

    def forward(self, arg_comp_a:ArgComp, arg_comp_b):


