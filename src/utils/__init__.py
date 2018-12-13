# read conf file
import os
from enum import Enum


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
        self.models_dir = os.path.join(hp_dict['base_dir'],"models")
        self.data_dir = os.path.join(hp_dict['base_dir'],"data")
        self.vocab_dir = os.path.join(self.data_dir,"vocabularies")
        self.exps_dir = os.path.join(hp_dict['base_dir'],"exps")
        self.rand_seed = int(hp_dict['rand_seed'])

class DivisionResolution(Enum):
    SENTENCE = 0
    PARAGRAPH = 1
    ESSAY = 2


mode_dict = {'s': DivisionResolution.SENTENCE,
             'p': DivisionResolution.PARAGRAPH,
             'e': DivisionResolution.ESSAY}