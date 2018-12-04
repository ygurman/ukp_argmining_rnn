import sys
from argparse import ArgumentParser
from enum import Enum

from src.models import BiLSTM_Segmentor_Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

class DivisionResolution(Enum):
    SENTENCE = 0
    PARAGRAPH = 1
    ESSAY = 2

def prepare_training_data(division_type, word2ix, pos2ix, tag2ix):

    indexed_tokens = []
    indexed_POSs = []
    indexed_AC_tags = []

    return indexed_tokens, indexed_POSs, indexed_AC_tags


def main():
    model = BiLSTM_Segmentor_Classifier(d_word_embd, d_pos_embd,
                 d_h1, n_lstm_layers, ac_tagset_size,
                 len_seq, pretraind_embd_layer=None)
    optimizer = optim.Adam(lr = args.learning_rate,weight_decay=args.wd)


    ## set CUDA if available
    if torch.cuda.is_available():
        torch.cuda.seed(361)

    for epoch in range(n_epochs):
        for indexed_tokens, indexed_POSs, indexed_tags in prepare_training_data():
            pass


if __name__ = "__main__":
    parser = ArgumentParser()
    args = parser.parse_args(sys.argv[1:])

    main(args)
