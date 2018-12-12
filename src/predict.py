# prediction using the segmentor-classifier for ACs

import sys
from argparse import ArgumentParser
from enum import Enum

import os
import torch


class DivisionResolution(Enum):
    SENTENCE = 0
    PARAGRAPH = 1
    ESSAY = 2


mode_dict = {'s': DivisionResolution.SENTENCE,
             'p': DivisionResolution.PARAGRAPH,
             'e': DivisionResolution.ESSAY}


# read cond file
class HyperParams(object):
    def __init__(self, conf_file_path):
        hp_dict = dict()
        with open(conf_file_path, 'rt') as f:
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
        self.batch_size = int(hp_dict['batch_size'])
        self.use_pos = True if hp_dict['use_pos'] == "True" else False
        # training and optimization parameters
        self.clip_threshold = int(hp_dict['clip_threshold'])
        self.learning_rate = float(hp_dict['learning_rate'])
        self.weight_decay = float(hp_dict['weight_decay'])
        self.n_epochs = int(hp_dict['n_epochs'])
        # general parameters
        self.models_dir = os.path.abspath(hp_dict['models_dir'])
        self.rand_seed = int(hp_dict['rand_seed'])


def main(mode, config_file_path, trained_model_path):
    # train the segmentor-classifier first
    h_params = HyperParams(config_file_path)
    from src.preprocess import get_train_test_split
    from src.preprocess import prepare_data
    from src.models import BiLSTM_Segmentor_Classifier
    from src.models import BiLSTM_Segmentor_Classifier_no_pos

    torch.manual_seed(h_params.rand_seed)

    _, test_files = get_train_test_split(os.path.abspath(os.path.join("..", "data", "train-test-split.csv")))
    test_data = prepare_data(mode, test_files)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SegmentorClassifier = BiLSTM_Segmentor_Classifier if h_params.use_pos else BiLSTM_Segmentor_Classifier_no_pos
    model = SegmentorClassifier(h_params.d_word_embd, h_params.d_pos_embd, h_params.d_h1,
                                h_params.n_lstm_layers, h_params.word_voc_size, h_params.pos_voc_size,
                                h_params.ac_tagset_size, h_params.batch_size, device,
                                h_params.pretraind_embd_layer_path)

    # load trained model state-dict
    checkpoint = torch.load(trained_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    ## set CUDA if available
    if torch.cuda.is_available():
        model.cuda()

    # set evaluation mode mode
    model.eval()

    # inference for all chosen data
    correct = 0
    total = 0
    with torch.no_grad():
        for (indexed_tokens, indexed_POSs, indexed_AC_tags) in test_data:
            tag_scores = model((indexed_tokens.to(device),indexed_POSs.to(device))) # get log soft max for input
            preds = torch.argmax(tag_scores, dim=1)

    print("{:2f}".format(100 * float(correct) / total ))
    # save results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', default='s', choices=['s', 'p', 'e'], help="""context learning mode:
        - 's' - sentence"
        - 'p' - paragraph"
        - 'e' - essay""")

    parser.add_argument('-cp', '--config_path', default=os.path.abspath(os.path.join("..", "params.conf")),
                        help=" path to learning parameters file")

    parser.add_argument('-mp', '--model_path', required=True, help=" path to trained model")
    args = parser.parse_args(sys.argv[1:])
    mode = mode_dict[args.mode]
    main(mode, os.path.abspath(args.config_path), os.path.abspath(args.model_path))
