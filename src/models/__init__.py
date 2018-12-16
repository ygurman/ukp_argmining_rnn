## model package for the 2-part tagger - (biLSTM AC segmentor-classifier & encoder-decoder relation classifier)
import pickle

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_Segmentor_Classifier(nn.Module):
    """
    BiLSTM based AC segmentor-Classifier. 1st part of the e2e model
    """

    def __init__(self, d_word_embd, d_pos_embd,
                 d_h1, n_lstm_layers, word_voc_size,
                 pos_voc_size, ac_tagset_size, batch_size,
                 device, pretraind_embd_layer_path=None):
        super().__init__()
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

        # initialize first hidden states
        self.hidden1 = self.init_hidden(self.h1dimension)

        # set hidden linear layer (for classification) that maps hidden states to AC type tags
        self.hidden2AC_tag = nn.Linear(in_features=d_h1 * 2, out_features=ac_tagset_size)

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
        :param prepared_sequence: tuple of 3 indexed lists (created via "prepare sequence method) - indexed tokens and POSs
        """
        indexed_tokens, indexed_POSs = prepared_sequence
        # pass word and pos indices as input
        w_embd = self.embd_word_layer(indexed_tokens)
        pos_embd = self.embd_pos_layer(indexed_POSs)
        # concatenate pos and word embeddings
        embd_output = torch.cat((w_embd, pos_embd), -1)

        # pass embedding layer output to lstm
        lstm_output, self.hidden1 = self.lstm1(embd_output.view(embd_output.size(0), self.batch_size, -1), self.hidden1)
        # pass lstm output to hidden liner layer
        tag_space = self.hidden2AC_tag(lstm_output.view(len(prepared_sequence[0]), -1))
        tag_scores = F.log_softmax(tag_space, dim=-1)
        return tag_scores


class BiLSTM_Segmentor_Classifier_no_pos(nn.Module):
    """
    BiLSTM based AC segmentor-Classifier. 1st part of the e2e model
    """

    def __init__(self, d_word_embd, d_pos_embd,
                 d_h1, n_lstm_layers, word_voc_size,
                 pos_voc_size, ac_tagset_size, batch_size,
                 device, pretraind_embd_layer_path=None):
        super().__init__()
        self.device = device
        self.h1dimension = d_h1
        self.batch_size = batch_size

        # embedding layers (words and POS tags)
        self.embd_word_layer = pickle.load(open(pretraind_embd_layer_path, 'rb')) if pretraind_embd_layer_path \
            else nn.Embedding(num_embeddings=word_voc_size, embedding_dim=d_word_embd)

        # sequence (bilstm) layer - recive previous concatenated embeddings and pass to linear classification (for tagging)
        self.lstm1 = nn.LSTM(input_size=d_word_embd,  # input as concatenated word|pos embeddings
                             hidden_size=d_h1,
                             # hidden state parameter (for token level AC class) to pass to the final linear layer
                             num_layers=n_lstm_layers,
                             # number of LSTM layers (set to 1 but passed as parameter for reusability)
                             bidirectional=True)  # sets as bi-drectional LSTM

        # initialize first hidden states
        self.hidden1 = self.init_hidden(self.h1dimension)

        # set hidden linear layer (for classification) that maps hidden states to AC type tags
        self.hidden2AC_tag = nn.Linear(in_features=d_h1 * 2, out_features=ac_tagset_size)

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
        :param prepared_sequence: tuple of 3 indexed lists (created via "prepare sequence method) - indexed tokens and POSs
        """
        indexed_tokens, _ = prepared_sequence
        # pass word and pos indices as input
        w_embd = self.embd_word_layer(indexed_tokens)

        # pass embedding layer output to lstm
        lstm_output, self.hidden1 = self.lstm1(w_embd.view(w_embd.size(0), self.batch_size, -1), self.hidden1)
        # pass lstm output to hidden liner layer
        tag_space = self.hidden2AC_tag(lstm_output.view(len(prepared_sequence[0]), -1))
        tag_scores = F.log_softmax(tag_space, dim=-1)
        return tag_scores

class BiLSTMRelationClassifier(nn.Module):
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
        self.rel_tagset_size = rel_tagset_size
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

        # 2 another biLSTM learning representation of the concatinated ACs using the new embedded features
        self.lstm2 = nn.LSTM(input_size=d_h1 * 2 + d_tag_embd + d_small_embd * 2 + d_distance_embd, hidden_size=d_h2,
                             num_layers=n_lstm_layers, bidirectional=True)
        self.hidden2 = self.init_hidden(self.h2dimension)
        self.fc1 = nn.Linear(in_features=2 * d_h2, out_features=d_h3)
        self.fc2 = nn.Linear(in_features=d_h3,out_features=rel_tagset_size)

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
        a_w_embd = self.embd_word_layer(ac_a.tokens.to(self.device))
        b_w_embd = self.embd_word_layer(ac_b.tokens.to(self.device))
        a_pos_embd = self.embd_pos_layer(ac_a.poss.to(self.device))
        b_pos_embd = self.embd_pos_layer(ac_b.poss.to(self.device))
        # concatenate pos and word embeddings
        a_embd_output = torch.cat((a_w_embd, a_pos_embd), -1)
        b_embd_output = torch.cat((b_w_embd, b_pos_embd), -1)

        # pass embedding layer output to lstm
        a_lstm_output, self.hidden1 = self.lstm1(a_embd_output.view(a_embd_output.size(0), self.batch_size, -1), self.hidden1)
        b_lstm_output, self.hidden1 = self.lstm1(b_embd_output.view(b_embd_output.size(0), self.batch_size, -1),self.hidden1)

        a_len = len(a_lstm_output)
        b_len = len(b_lstm_output)

        ## part 2 -linear combinations
        # ac features - duplicate and concat to appropriate ac
        a_embd_ac_tag_layer = self.embd_ac_tag_layer(torch.tensor(ac_a.type).to(self.device))
        b_embd_ac_tag_layer = self.embd_ac_tag_layer(torch.tensor(ac_b.type).to(self.device))

        # append tag embeddings to each component
        a = torch.cat([a_lstm_output.view(a_len,-1),a_embd_ac_tag_layer.repeat(a_len,1)],dim=-1)
        b = torch.cat([b_lstm_output.view(b_len,-1),b_embd_ac_tag_layer.repeat(b_len,1)],dim=-1)

        # create combined representation (by concatination)
        ab = torch.cat([a,b],dim=0)

        # relation features - duplicate and concat to entire
        embd_a_before_b = self.embd_a_is_before(torch.tensor(ac_a.before(ac_b),dtype=torch.long).to(self.device))
        embd_same_par = self.embd_in_same_par(torch.tensor(ac_a.same_paragraph(ac_b),dtype=torch.long).to(self.device))
        embd_dist = self.embd_distance(torch.tensor(ac_a.distance(ac_b),dtype=torch.long).to(self.device))

        combined_embds = torch.cat([embd_a_before_b,embd_same_par,embd_dist],dim=-1)

        # add constructed features embeddings to representation (size (len_a+len_b)*(hidden1+all the embeddings))
        ab = torch.cat([ab,combined_embds.repeat(a_len+b_len,1)],dim=-1)

        # pass through second lstm with new features
        lstm2_out, self.hidden2 = self.lstm2(ab.view(a_len+b_len,self.batch_size,-1),self.hidden2)

        # pass through second linear layer with ReLU activation
        ab = F.relu(self.fc1(lstm2_out.view(len(ab),-1)))

        # flatten (by mean) to hidden dimension and pass through last liner layer for mapping with logsoftmax
        ab = F.relu(self.fc2(ab))
        ab = ab.sum(dim=0)

        tag_scores = F.log_softmax(ab,dim=0).view(1,-1)

        return tag_scores

class BlandRelationClassifier(nn.Module):
    """
    a leaner model for relation classification w/o constructed parameters embeddings
    """
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
        self.rel_tagset_size = rel_tagset_size
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

        # another biLSTM learning representation of the concatinated ACs using the new embedded features
        self.lstm2 = nn.LSTM(input_size=d_h1 , hidden_size=d_h2,num_layers=n_lstm_layers, bidirectional=True)
        self.hidden2 = self.init_hidden(self.h2dimension)
        self.fc1 = nn.Linear(in_features=2 * d_h2, out_features=d_h3)
        self.fc2 = nn.Linear(in_features=d_h3,out_features=rel_tagset_size)

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

        # pass through second lstm with new features
        lstm2_out, self.hidden2 = self.lstm2(ab.view(ab.size(0),self.batch_size,-1),self.hidden2)

        # pass through second linear layer with ReLU activation
        ab = F.relu(self.fc1(lstm2_out.view(len(ab),-1).sum(dim=0)))

        # flatten (by summation) to hidden dimension and pass through last liner layer for mapping with logsoftmax
        ab = ab.sum(dim=0)

        tag_scores = F.log_softmax(ab.view(self.rel_tagset_size,-1))

        return tag_scores

class BaselineRelationClassifier(nn.Module):
    """
    a leaner model for relation classification w/o constructed parameters embeddings
    """
    def __init__(self, d_word_embd, d_pos_embd,
                 d_h1, n_lstm_layers, word_voc_size,
                 pos_voc_size, ac_tagset_size, batch_size,
                 device, pretraind_embd_layer_path=None,rel_tagset_size=9, d_tag_embd=25,
                 d_small_embd=5,d_distance_embd=15, d_h2 = 50, d_h3=25):
        super().__init__()

        ## part 1 - using pretrained segmentor classifier up until LSTM layer representation
        self.device = device
        self.h1dimension = d_h1
        self.rel_tagset_size = rel_tagset_size
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

        # set the first part as static (non trainablewise)
        for layer in [self.embd_word_layer, self.embd_pos_layer, self.lstm1]:
            for parameter in layer.parameters():
                parameter.requires_grad=False

        # part 2 - relation classification extraction
        # simple linear layer using flattened represntation od the parameters
        self.fc = nn.Linear(in_features=2 * self.h1dimension, out_features=rel_tagset_size)

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
        a_w_embd = self.embd_pos_layer(ac_a.tokens.to(self.device))
        b_w_embd = self.embd_word_layer(ac_b.tokens.to(self.device))
        a_pos_embd = self.embd_pos_layer(ac_a.poss.to(self.device))
        b_pos_embd = self.embd_pos_layer(ac_b.poss.to(self.device))
        # concatenate pos and word embeddings
        a_embd_output = torch.cat((a_w_embd, a_pos_embd), -1)
        b_embd_output = torch.cat((b_w_embd, b_pos_embd), -1)

        # pass embedding layer output to lstm
        a_lstm_output, self.hidden1 = self.lstm1(a_embd_output.view(a_embd_output.size(0), self.batch_size, -1), self.hidden1)
        b_lstm_output, self.hidden1 = self.lstm1(a_embd_output.view(b_embd_output.size(0), self.batch_size, -1),self.hidden1)

        a_len = len(a_lstm_output)
        b_len = len(b_lstm_output)

        ## part 2 -linear combinations
        # flatten representation to hidden sizes and multiply to get relations
        a = a_lstm_output.view(a_len,-1).mean(dim=0)
        b = a_lstm_output.view(b_len, -1).mean(dim=0)
        ab = a * b # size [hidden * 2,1]

        # linear layer to tagset space
        tags_space = self.fc(ab)
        tag_scores = F.log_softmax(tags_space)

        return tag_scores

class BaselineConstructedRelationClassifier(nn.Module):
    """
    a leaner model for relation classification w/o constructed parameters embeddings
    """
    def __init__(self, d_word_embd, d_pos_embd,
                 d_h1, n_lstm_layers, word_voc_size,
                 pos_voc_size, ac_tagset_size, batch_size,
                 device, pretraind_embd_layer_path=None,rel_tagset_size=9, d_tag_embd=25,
                 d_small_embd=5,d_distance_embd=15, d_h2 = 50, d_h3=25):
        super().__init__()

        ## part 1 - using pretrained segmentor classifier up until LSTM layer representation
        self.device = device
        self.h1dimension = d_h1
        self.rel_tagset_size = rel_tagset_size
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

        # set the first part as static (non trainablewise)
        for layer in [self.embd_word_layer, self.embd_pos_layer, self.lstm1]:
            for parameter in layer.parameters():
                parameter.requires_grad=False

        # part 2 - relation classification extraction
        # simple linear layer using flattened represntation od the parameters
        self.embd_ac_tag_layer = nn.Embedding(num_embeddings=ac_tagset_size,embedding_dim=d_tag_embd)
        self.embd_a_is_before = nn.Embedding(num_embeddings=2,embedding_dim=d_small_embd) #boolen
        self.embd_in_same_par = nn.Embedding(num_embeddings=2,embedding_dim=d_small_embd) # boolean
        self.embd_distance = nn.Embedding(num_embeddings=30,embedding_dim=d_distance_embd) # bounded by the maximal id
        self.fc = nn.Linear(in_features=2 * self.h1dimension + d_tag_embd + d_distance_embd + 2 *d_small_embd, out_features=rel_tagset_size)

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
        a_w_embd = self.embd(ac_a.tokens.to(self.device))
        b_w_embd = self.embd_word_layer(ac_b.tokens.to(self.device))
        a_pos_embd = self.embd_pos_layer(ac_a.poss.to(self.device))
        b_pos_embd = self.embd_pos_layer(ac_b.poss.to(self.device))
        # concatenate pos and word embeddings
        a_embd_output = torch.cat((a_w_embd, a_pos_embd), -1)
        b_embd_output = torch.cat((b_w_embd, b_pos_embd), -1)

        # pass embedding layer output to lstm
        a_lstm_output, self.hidden1 = self.lstm1(a_embd_output.view(a_embd_output.size(0), self.batch_size, -1), self.hidden1)
        b_lstm_output, self.hidden1 = self.lstm1(a_embd_output.view(b_embd_output.size(0), self.batch_size, -1),self.hidden1)

        a_len = len(a_lstm_output)
        b_len = len(b_lstm_output)

        ## part 2 -linear combinations
        a = a_lstm_output.view(a_len, -1).mean(dim=0)
        b = a_lstm_output.view(b_len, -1).mean(dim=0)

        # constructed features
        # ac features - duplicate and concat to appropriate ac
        a_embd_ac_tag_layer = self.embd_ac_tag_layer(torch.tensor(ac_a.type).to(self.device))
        b_embd_ac_tag_layer = self.embd_ac_tag_layer(torch.tensor(ac_b.type).to(self.device))

        # append tag embeddings to each component
        a = torch.cat([a_lstm_output.view(a_len, -1), a_embd_ac_tag_layer.repeat(a_len, 1)], dim=-1)
        b = torch.cat([b_lstm_output.view(b_len, -1), b_embd_ac_tag_layer.repeat(b_len, 1)], dim=-1)

        # relation features - duplicate and concat to entire
        embd_a_before_b = self.embd_a_is_before(torch.tensor(ac_a.before(ac_b), dtype=torch.long).to(self.device))
        embd_same_par = self.embd_in_same_par(torch.tensor(ac_a.same_paragraph(ac_b), dtype=torch.long).to(self.device))
        embd_dist = self.embd_distance(torch.tensor(ac_a.distance(ac_b), dtype=torch.long).to(self.device))

        combined_embds = torch.cat([embd_a_before_b, embd_same_par, embd_dist], dim=-1)

        # add constructed features embeddings to representation (size (len_a+len_b)*(hidden1+all the embeddings))
        a = torch.cat([a,combined_embds.repeat(a_len,1)],dim=-1)
        b = torch.cat([b, combined_embds.repeat(b_len, 1)], dim=-1)

        # flatten representation to hidden sizes and multiply to get relations
        ab = a * b # size [(hidden * 2 + 2*d_embd_small +d_embd_tag + d_embd_dist),1]

        # linear layer to tagset space
        tags_space = self.fc(ab)
        tag_scores = F.log_softmax(tags_space)

        return tag_scores
