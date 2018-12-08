## model package for the 2-part tagger - (biLSTM AC segmentor-classifier & encoder-decoder relation classifier)
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class BiLSTM_Segmentor_Classifier(nn.Module):
    """
    BiLSTM based AC segmentor-Classifier. 1st part of the e2e model
    """

    def __init__(self, d_word_embd, d_pos_embd,
                 d_h1, n_lstm_layers, word_voc_size,
                 pos_voc_size, ac_tagset_size, pretraind_embd_layer_path=None):
        super().__init__()
        self.h1dimension = d_h1

        # embedding layers (words and POS tags)
        self.embd_word_layer = pickle.load(open(pretraind_embd_layer_path, 'rb')) if pretraind_embd_layer_path \
            else nn.Embedding(num_embeddings=word_voc_size, embedding_dim=d_word_embd)
        self.embd_pos_layer - nn.Embedding(num_embeddings=pos_voc_size, embedding_dim=d_pos_embd)

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
        h0 = autograd.Variable(torch.zeros(1 * 2, self.batch_size, hidden_dim))
        c0 = autograd.Variable(torch.zeros(1 * 2, self.batch_size, hidden_dim))
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


def main():
    # prepare_train_data()
    pass


if __name__ == "__main__":
    main()

# TODO - for relation segmentation use structered flags (in same paragraph, a before b, a/b type) along with tokens representation from first Model
