## model package for the 2-part tagger - (biLSTM AC segmentor-classifier & encoder-decoder relation classifier)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import pickle
import sys

torch.manual_seed(361)


# utility function for input preparation
def prepare_sequence(seq:[str], to_ix:dict) -> torch.tensor:
    """
    use pre-defined indexed vocabulary and return a tensor of integers appropriate to string input (list of tokens/POSs/tags)
    used as input for the segmentor/classifier (for the first embedding layers)
    :param seq: list of tokens/POSs/tags
    :param to_ix: dictionary incexing the tokens/POSs/tags vocabulary
    :return: matching torch.tensor of integers representing input lists
    """
    idxs = [to_ix[w] for w in seq] # replace input tokens/POSs with matching indices
    return torch.tensor(idxs, dtype=torch.long) # convert to torch tensor type

def build_vocabularies(data_path,train_file_names):
    """
    read files in order to create vocabularies of tags (AC types and rel Types) and inputs (words, POSs)
    :param data_path: base data directory
    :param train_file_names: list of essays in the train set (following the "essay{3d}.tsv" convention)
    :return: save the vocabularies for later use
    """
    word_voc = set()
    pos_voc = set()
    ac_tag_voc = set()
    ac_rel_voc = set()

    # iterate over the pre-processed conll-like data to build vocabulary
    for essay in train_file_names:
        essay_path = os.path.join(data_path,"data","processed",essay + ".tsv")
        with open(essay_path,"rt") as f:
            for line in f:
                if line[0] == "#":
                    continue
                tok, pos, ac_tag, _, rel_tag = line.split('\t')
                word_voc.add(tok.lower())
                pos_voc.add(pos)
                ac_tag_voc.add(ac_tag)
                ac_rel_voc.add(rel_tag.split(":")[0])

    # build indexed dictionaries for the vocabularies
    word2ix = dict((w, i) for i,w in enumerate(word_voc))
    pos2ix = dict((w, i) for i, w in enumerate(pos_voc))
    ac_tag2ix = dict((w, i) for i, w in enumerate(ac_tag_voc))
    rel_2ix = dict((w, i) for i, w in enumerate(ac_rel_voc))

    # save the vocabs to processed data folder for later use
    for name, dic in (("word_2ix", word2ix), ("pos2ix", pos2ix), ("ac_tag2ix", ac_tag2ix), ("ac_rel2ix", rel_2ix)):
        pickle.dump(dic,open(os.path.join(data_path,"processed",name + ".pcl"),"wb"))

    return word2ix, pos2ix, ac_tag2ix, rel_2ix

# use the original train-test split supplied with the UKP dataset
def get_train_test_split(split_path) -> ([str] , [str]):
    """
    takes the semi-colon delimited original UKP dataset train-test cut and returns list of test and train filenames
    :param split_path:
    :return:
    """
    train_set = []
    test_set = []
    with open(split_path) as f:
        f.readline() # skip the first line
        for line in f:
            name, set = line.strip().replace("\"","").split(";")
            if set == "TRAIN":
                train_set.append(name)
            else:
                test_set.append(name)
    return train_set, test_set

def prepare_glove_embeddings(glove_embds_path, glove_txt_name):
    """
    create a vocabulary and numpy vectors from GloVe txt file and create and store in appropriate pickle file
    :param glove_embds_path:
    :return:
    """
    words = []
    ix = 0
    word2ix = dict()
    vectors = []

    sys.stdout.write("loading GloVe vectors\n")
    with open(os.path.join(glove_embds_path,glove_txt_name),'rt') as f:
        for line in f:
            split = line.split()
            word = split[0]
            embd = np.array([float(val) for val in split[1:]])

            words.append(word)
            word2ix[word] = ix
            ix += 1
            vectors.append(embd)

    sys.stdout.write("saving files to {}\n".format(glove_embds_path))
    pickle.dump(words,open(os.path.join(glove_embds_path,glove_txt_name.replace(".txt","_words.pcl")),"wb"))
    pickle.dump(word2ix, open(os.path.join(glove_embds_path, glove_txt_name.replace(".txt", "_word2ix.pcl")), "wb"))

    glove_embds = {w: vectors[word2ix[w]] for w in words}
    pickle.dump(glove_embds, open(os.path.join(glove_embds_path, glove_txt_name.replace(".txt", "_vectors.pcl")), "wb"))

    sys.stdout.write("files saved to {}\n".format(glove_embds_path))

    return words, word2ix, glove_embds

def combine_word_vocab(dataset_voc, pretrained_embs, d_embd):
    """
    combine dataset's vocabulary with GloVe to create an embedding layer for words (initiating unkown tokens with random normal distributed weights)
    :param dataset_voc:
    :param pretrained_embs_voc:
    :return: vocabulary_size, embeddings_dimentions, embedding layer with appropriate weights
    """
    new_word2ix = {w: i for (i,w) in enumerate(pretrained_embs.keys())}
    new_weights = dict()
    new_ix = len(new_word2ix.keys())

    diff = 0
    for ds_word in dataset_voc:
        if ds_word not in pretrained_embs.keys():
            diff += 1
            # randomize normal distributed weight for unknown word
            weight = np.random.normal(scale=0.5,size=(d_embd,)) # TODO - consider changing 'scale' parameter
            new_word2ix[ds_word] = new_ix
            new_weights[new_ix] = weight
            new_ix += 1

    # create weight matrix
        voc_size = len(new_word2ix.keys())
    weight_matrix = np.zeros((voc_size,d_embd))

    for word, ix in new_word2ix.items():
        try:
            weight_matrix[i] = pretrained_embs[word]
        except KeyError:
            weight_matrix[i] = new_weights[i]

    # create embedding layer
    embd_layer = nn.Embedding(num_embeddings=voc_size,embedding_dim=d_embd)
    embd_layer.load_state_dict({'weight':weight_matrix})

    return new_word2ix, embd_layer

# actual model
class BiLSTM_Segmentor_Classifier(nn.Module):
    def __init__(self, d_word_embd, d_pos_embd, d_h1, n_lstm_layers, ac_tagset_size, len_seq, pretraind_embd_layer=None):
        super().__init__()
        self.h1dimension = d_h1

        # embedding layers (words and POS tags)
        self.embd_words = pretraind_embd_layer or nn.Embedding(num_embeddings=len_seq,embedding_dim=d_word_embd)
        self.emd_pos - nn.Embedding(num_embeddings=len_seq,embedding_dim=d_pos_embd)

        # sequence (bilstm) layer - recive previous concatenated embeddings and pass to linear classification (for tagging)
        self.seq_layer = nn.LSTM(input_size = d_word_embd + d_pos_embd, # input as concatenated word|pos embeddings
                                 hidden_size = d_h1,                    # hidden state parameter (for token level AC class) to pass to the final linear layer
                                 num_layers = n_lstm_layers,            # number of LSTM layers (set to 1 but passed as parameter for reusability)
                                 bidirectional = True)                  # sets as bi-drectional LSTM

        # initialize first hidden states
        self.hidden1 = init_hidden(self.h1dimension)

        # set hidden linear layer (for classification) that maps hidden states to AC type tags
        self.hidden2AC_tag = nn.Linear(in_features = d_h1, out_features = ac_tagset_size)

    def init_hidden(self, hidden_dim):
        # initialize hidden states with zeroes (2 because bidirectional lstm)
        h0 = autograd.Variable(torch.zeros(1 * 2, self.batch_size, hidden_dim))
        c0 = autograd.Variable(torch.zeros(1 * 2, self.batch_size, hidden_dim))
        return h0, c0

    # def init_word_embeddings(self.pre_trained_embeddings_path):
    #     pass # TODO


    def forward():
        ## AC-type recognition side
        # pass word and pos indices as input
        w_embd = self.embd_word_layer(word)
        pos_embd = self.embd_pos_layer(pos)
        # concatenate pos and word embeddings
        embd_output = torch.cat((w_embd, pos_embd), -1)

        # pass embedding layer output to lstm
        lstm_output, self.hidden1 = self.lstm1_ent(embd_output.view(self.h1dimension, self.batch_size, -1), self.hidden1)
        # pass lstm output to hidden liner layer
        tag_space = self.hidden2AC_tag(lstm_output.view(-1,lstm_output.size(-1)))
        tag_scores = F.log_softmax(tag_space, dim=-1)
        return tag_scores

def main():
    pass

if __name__ == "__main__":
    main()