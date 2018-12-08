import sys
import time
from argparse import ArgumentParser
from enum import Enum

import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


class DivisionResolution(Enum):
    SENTENCE = 0
    PARAGRAPH = 1
    ESSAY = 2

mode_dict = {'s' : DivisionResolution.SENTENCE,
             'p' : DivisionResolution.PARAGRAPH,
             'e' : DivisionResolution.ESSAY}

# read cond file
class HyperParams(object):
    def __init__(self, conf_file_path):
        hp_dict = dict()
        with open(conf_file_path,'rt') as f:
            for line in f:
                if line[0] != '#':
                    prop, val = line.strip().split(": ")
                    hp_dict[prop] = val
        # segmentor-classifier parameters
        self.d_word_embd = int(hp_dict["d_word_embd"])
        self.d_pos_embd = int(hp_dict["d_pos_embd"])
        self.n_lstm_layers = int(hp_dict["n_lstm_layers"])
        self.d_h1 = int(hp_dict['d_h1'])
        self.word_voc_size = int(hp_dict["word_voc_size"])
        self.pos_voc_size = int(hp_dict["pos_voc_size"])
        self.ac_tagset_size = int(hp_dict["ac_tagset_size"])
        self.pretraind_embd_layer_path = os.path.abspath(hp_dict['pretraind_embd_layer_path'])
        # training and optimization parameters
        self.clip_threshold = int(hp_dict['clip_threshold'])
        self.learning_rate = float(hp_dict['learning_rate'])
        self.weight_decay = float(hp_dict['weight_decay'])
        self.n_epochs = int(hp_dict['n_epochs'])
        # general parameters
        self.models_dir = os.path.abspath(hp_dict['models_dir'])


def main(mode, config_file_path):
    # train the segmentor-classifier first
    h_params = HyperParams(config_file_path)
    from src.preprocess import get_train_test_split
    from src.preprocess import prepare_data
    from src.models import BiLSTM_Segmentor_Classifier

    training_files, _ = get_train_test_split(os.path.abspath(os.path.join("..","data","train-test-split.csv")))
    training_data = prepare_data(mode,training_files)

    model = BiLSTM_Segmentor_Classifier(h_params.d_word_embd, h_params.d_pos_embd, h_params.d_h1,
                                        h_params.n_lstm_layers, h_params.word_voc_size, h_params.pos_voc_size,
                                        h_params.ac_tagset_size, h_params.pretraind_embd_layer_path)

    # set loss function and adam optimizer (using negative log likelihood with adam optimizer
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=h_params.learning_rate, weight_decay=h_params.weight_decay)

    ## set CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.cuda()
        loss_function.cuda()

    # display parameters in model
    params = list(model.parameters()) + list(loss_function.parameters())
    total_params = sum(
        x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    sys.stdout.write('Args:{}\n'.format(params))
    sys.stdout.write('Model total parameters:{}\n'.format(total_params))

    # set train mode
    model.train()

    for epoch in range(h_params.n_epochs):
        start_time = time.time()
        acc_loss = 0.0 # accumalating loss per epoch for display
        for ((indexed_tokens,indexed_POSs), indexed_AC_tags) in tqdm(training_data):
            # reset accumalated gradients and lstm's hidden state between iterations
            model.zero_grad()
            model.hidden1 = model.init_hidden(model.h1dimension)

            # make a forward pass
            tag_scores = model((indexed_tokens.to(device),indexed_POSs.to(device)))

            # backprop
            loss = loss_function(tag_scores,indexed_AC_tags)
            acc_loss += loss.item()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(),h_params.clip_threshold)

            # call optimizer step
            optimizer.step()
        end_time = time.time()
        # output stats
        sys.stdout.write("===> Epoch[{}/{}]: Loss: {:.4f} , time = {:d}[s]\n".format(epoch, h_params.n_epochs,acc_loss,int(end_time-start_time)))

        if epoch in [25,50,75]:
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, os.path.abspath(os.path.join(h_params.models_dir, "SegClass_mode-{}_ep-{}.pt".format(mode,epoch))))
            except:
                sys.stdout.write('failed to save model in epoch {}\n'.format(epoch))

    # save model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, os.path.abspath(os.path.join(h_params.models_dir,"SegClass_{}.pt".format(mode))))

    #announce end
    sys.stdout.write("finished training")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m','--mode', default='s', choices=['s','p','e'],help= """context learning mode:
        - 's' - sentence"
        - 'p' - paragraph"
        - 'e' - essay""")

    parser.add_argument('-cp','--config_path', default=os.path.abspath(os.path.join("..","params.conf")),
                        help= " path to learning parameters file")
    args = parser.parse_args(sys.argv[1:])
    mode = mode_dict[args.mode]
    main(mode, os.path.abspath(args.config_path))
