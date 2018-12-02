## model package for the 2-part tagger - (biLSTM AC segmentor-classifier & encoder-decoder relation classifier)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import os
import numpy as np
import pickle
import sys
from typing import Dict, List

UNK_TOKEN_SYMBOL = "UNKNOWN_TOKEN"
PAD_SYMBOL = "PAD_SYM"
torch.manual_seed(361)
np.random.seed(361)

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
        essay_path = os.path.join(data_path,"processed",essay + ".tsv")
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
    voc_dir = os.path.abspath(os.path.join("..","data","vocabularies"))
    if not os.path.exists(voc_dir):
        os.mkdir(voc_dir)
    for name, dic in (("word_2ix", word2ix), ("pos2ix", pos2ix), ("ac_tag2ix", ac_tag2ix), ("ac_rel2ix", rel_2ix)):
        pickle.dump(dic,open(os.path.join(data_path,"vocabularies",name + ".pcl"),"wb"))

    sys.stdout.write("wrote vocabularies to {}\n".format(os.path.abspath(voc_dir)))
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
    pickle.dump(words,open(os.path.join(glove_embds_path,glove_txt_name.replace(".txt","_words_lst.pcl")),"wb"))
    pickle.dump(word2ix, open(os.path.join(glove_embds_path, glove_txt_name.replace(".txt", "_word2ix_dict.pcl")), "wb"))

    glove_embds = {w: vectors[word2ix[w]] for w in words}
    pickle.dump(glove_embds, open(os.path.join(glove_embds_path, glove_txt_name.replace(".txt", "_word2vectors_dict.pcl")), "wb"))

    sys.stdout.write("files saved to {}\n".format(glove_embds_path))

    return words, word2ix, glove_embds


def combine_word_vocab(dataset_voc:Dict[str,int], pretrained_embds:Dict[str,np.array], d_embd:int, save_path="."):
    """
    combine dataset's vocabulary with GloVe to create an embedding layer for words (initiating unkown tokens with random normal distributed weights)
    :param dataset_voc: a word2ix dictionary
    :param pretrained_embs_voc:
    :return: vocabulary_size, embeddings_dimentions, embedding layer with appropriate weights
    """
    new_word2ix = {w: i for (i,w) in enumerate(pretrained_embds.keys(),1)} #save index 0 for padding (if implementing batching
    new_word2ix[PAD_SYMBOL] = 0 # reserve un-initalized weight vector for padding symbol

    new_weights = dict()
    new_ix = len(new_word2ix.keys())

    diff = 0

    std = 1/np.sqrt(d_embd) # standard deviation for randomized vectors

    for ds_word in dataset_voc.keys():
        if ds_word not in pretrained_embds.keys():
            diff += 1
            # randomize normal distributed weight for unknown word using Xavier's initialization variant
            weight = np.random.normal(0, scale=std, size=(d_embd,)).astype(np.float32)
            new_word2ix[ds_word] = new_ix
            new_weights[new_ix] = weight
            new_ix += 1

    # add UNKNOWN_TOKEN_SYMBOL for later use
    new_word2ix[UNK_TOKEN_SYMBOL] = new_ix
    new_weights[new_ix] = np.random.normal(0, scale=std, size=(d_embd,))
    new_weights[new_word2ix[PAD_SYMBOL]] = np.zeros(d_embd,)

    # create weight matrix
    voc_size = len(new_word2ix.keys())

    weight_matrix = np.zeros((voc_size,d_embd))

    for word, ix in new_word2ix.items():
        try:
            weight_matrix[ix] = pretrained_embds[word]
        except KeyError:
            weight_matrix[ix] = new_weights[ix]

    # create embedding layer
    embd_layer = nn.Embedding(num_embeddings=voc_size,embedding_dim=d_embd)
    embd_layer.weight.data = torch.Tensor(weight_matrix)

    if save_path:
        with open(os.path.join(save_path,"combined_word_voc_word2ix.pcl"), "wb") as f:
            pickle.dump(obj=new_word2ix,file=f)
        with open(os.path.join(save_path,"combined_word_voc_embdLayer.pcl"), "wb") as f:
            pickle.dump(obj=embd_layer,file=f)
        sys.stdout.write("\n")

    sys.stdout.write("saved combined vocab and embedding layer to {}\n".format(save_path))

    return new_word2ix, embd_layer


# actual model
class BiLSTM_Segmentor_Classifier(nn.Module):
    """
    BiLSTM based AC segmentor-Classifier. 1st part of the e2e model
    """
    def __init__(self, d_word_embd, d_pos_embd,
                 d_h1, n_lstm_layers, ac_tagset_size,
                 len_seq, pretraind_embd_layer=None):

        super().__init__()
        self.h1dimension = d_h1

        # embedding layers (words and POS tags)
        self.embd_words = pretraind_embd_layer or nn.Embedding(num_embeddings=len_seq,embedding_dim=d_word_embd)
        self.emd_pos - nn.Embedding(num_embeddings=len_seq,embedding_dim=d_pos_embd)

        # sequence (bilstm) layer - recive previous concatenated embeddings and pass to linear classification (for tagging)
        self.lstm1 = nn.LSTM(input_size = d_word_embd + d_pos_embd, # input as concatenated word|pos embeddings
                                 hidden_size = d_h1,                    # hidden state parameter (for token level AC class) to pass to the final linear layer
                                 num_layers = n_lstm_layers,            # number of LSTM layers (set to 1 but passed as parameter for reusability)
                                 bidirectional = True)                  # sets as bi-drectional LSTM

        # initialize first hidden states
        self.hidden1 = self.init_hidden(self.h1dimension)

        # set hidden linear layer (for classification) that maps hidden states to AC type tags
        self.hidden2AC_tag = nn.Linear(in_features = d_h1 * 2, out_features = ac_tagset_size)

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
        tag_space = self.hidden2AC_tag(lstm_output.view(len(prepared_sequence[0]),-1))
        tag_scores = F.log_softmax(tag_space, dim=-1)
        return tag_scores


def prepare_train_data():
    # get train-test split
    data_path = os.path.abspath(os.path.abspath(os.path.join("..", "data")))
    split_path = os.path.join(data_path, "train-test-split.csv")
    train_files, _ = get_train_test_split(split_path)
    # build and save vocabularies as word2index dictionaries
    word2ix, pos2ix, ac_tag2ix, rel_2ix = build_vocabularies(data_path=data_path, train_file_names=train_files)
    # prepare and save pre-trained GloVe word embeddings as word2np.array indices
    word_embd_dim = 100
    save_path = os.path.abspath(os.path.join(data_path, "vocabularies"))
    sys.stdout.write("processing pre-trained embeddings\n")
    _, _, glove_embds = prepare_glove_embeddings(glove_embds_path=os.path.join(data_path, "glove.6B"),
                                                 glove_txt_name="glove.6B.100d.txt")
    sys.stdout.write("building combined vocabulary and pretrained embedding layer\n")
    _, word_embd_layer = combine_word_vocab(dataset_voc=word2ix, pretrained_embds=glove_embds, d_embd=word_embd_dim,
                                            save_path=save_path)


def main():
    # prepare_train_data()
    pass

if __name__ == "__main__":
    main()