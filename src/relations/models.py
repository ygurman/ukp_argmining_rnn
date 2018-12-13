## model package for the 2-part tagger - (biLSTM AC segmentor-classifier & encoder-decoder relation classifier)
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from src.relations.preprocess import ArgComp


class blandRelationClassifier(nn.Module):
    """
    a lean linear model for relation classification consisting of ACs display using dotwise mean
    followed by matrix multiplication in order to create a 100x100 representation of every two ACs
    then through
    """

class biLSTMRelationClassifier(nn.Module):
    def __init__(self, d_word_embd, d_pos_embd,
                 d_h1, n_lstm_layers, word_voc_size,
                 pos_voc_size, ac_tagset_size, batch_size,
                 device, pretraind_embd_layer_path=None,rel_tagset_size=9, d_tag_embd=25,
                 d_small_embd=5,d_distance_embd=15, d_h2 = 50, d_h3=25):
        super().__init__()

        ## part 1 - using pretrained segmentor classifier up until LSTM layer representation
        self.device = device
        self.h1dimension = d_h1
        self.h2dimension = d_h2
        self.h3dimension = d_h3
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

        # initialize first hidden states
        self.hidden1 = self.init_hidden(self.h1dimension)

        # part 2 - relation classification extraction
        # embedding the constructed features in order to be able to learn their contribution to the prediction
        self.embd_ac_tag_layer = nn.Embedding(num_embeddings=ac_tagset_size,embedding_dim=d_tag_embd)
        self.embd_a_is_before = nn.Embedding(num_embeddings=2,embedding_dim=d_small_embd) #boolen
        self.embd_in_same_par = nn.Embedding(num_embeddings=2,embedding_dim=d_small_embd) # boolean
        self.embd_distance = nn.Embedding(num_embeddings=30,embedding_dim=d_distance_embd) # bounded by the maximal id

        # 2 feed forword sequences (regular linear layers with ReLUs and Dropout regularization
        self.fc1 = nn.Linear(in_features=d_h1+d_distance_embd+d_small_embd*2+d_distance_embd,out_features=d_h2)
        self.fc2 = nn.Linear(in_features=d_h2,out_features=d_h3)
        self.fc3 = nn.Linear(d_h3,rel_tagset_size)

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

    def forward(self, prepared_sequence):
        """
        :param prepared_sequence: tuple of prepared ACs as dicionaties
        """
        ## part 1 - old lstm representation
        # pass both to get lstm hidden representation of AC components
        ac_a, ac_b = prepared_sequence
        a_w_embd = self.embd(ac_a.tokens)
        b_w_embd = self.embd_word_layer(ac_b.tokens)
        a_pos_embd = self.embd_pos_layer(ac_a.poss)
        b_pos_embd = self.embd_pos_layer(ac_b.poss)
        # concatenate pos and word embeddings
        a_embd_output = torch.cat((a_w_embd, a_pos_embd), -1)
        b_embd_output = torch.cat((b_w_embd, b_pos_embd), -1)

        # pass embedding layer output to lstm
        a_lstm_output, self.hidden1 = self.lstm1(a_embd_output.view(a_embd_output.size(0), self.batch_size, -1), self.hidden1)
        b_lstm_output, self.hidden1 = self.lstm1(a_embd_output.view(b_embd_output.size(0), self.batch_size, -1),self.hidden1)

        a_len = len(a_lstm_output)
        b_len = len(b_lstm_output)

        ## part 2 -linear combinations
        # ac features - duplicate and concat to appropriate ac
        a_embd_ac_tag_layer = self.embd_ac_tag_layer(ac_a.type)
        b_embd_ac_tag_layer = self.embd_ac_tag_layer(ac_b.type)

        # append tag embeddings to each component
        a = torch.cat([a_lstm_output.view(a_len,-1),a_embd_ac_tag_layer.repeat(a_len,1)],dim=-1)
        b = torch.cat([b_lstm_output.view(b_len,-1),b_embd_ac_tag_layer.repeat(a_len,1)],dim=-1)

        # create combined representation (by concatination)
        ab = torch.cat([a,b],dim=0)
        ba = torch.cat([b,a],dim=0)

        # relation features - duplicate and concat to entire
        embd_a_before_b = self.embd_a_is_before(torch.tensor(ac_a.before(ac_b),dtype=torch.long).to(self.device))
        embd_same_par = self.embd_in_same_par(torch.tensor(ac_a.same_paragraph(ac_b),dtype=torch.long).to(self.device))
        embd_dist = self.embd_distance(torch.tensor(ac_a.distance(ac_b),dtype=torch.long).to(self.device))

        combined_embds = torch.cat([embd_a_before_b,embd_same_par,embd_dist],dim=-1)

        # add constructed features embeddings to representation (size (len_a+len_b)*(hidden1+all the embeddings))
        ab = torch.cat([ab,combined_embds.repeat(a_len,1)],dim=-1)
        ba = torch.cat([ba, combined_embds.repeat(a_len, 1)], dim=-1)

        # pass through linear layer
        ab = F.relu(self.fc1(ab))
        ba = F.relu(self.fc1(ba))

        # flatten the concatenated representation to the hidden dimension before next linear layer (now d_h2)
        ab = ab.sum(dim=0)
        ba = ba.sum(dim=0)

        # pass through another linear layer
        ab = F.relu(self.fc2(ab))
        ba = F.relu(self.fc2(ba))

        tag_space = self.fc3(ab+ba)
        tag_scores = F.log_softmax(tag_space)

        return tag_scores