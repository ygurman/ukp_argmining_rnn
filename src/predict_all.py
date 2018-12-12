# prediction using the segmentor-classifier for ACs
import pickle
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


def post_process(preds, ac_tag2ix):
    """
    inforce BIO restriction to improve classification (component begins with "B", type chosen by majority, only "I" within AC and no solitary ac's
    :param preds: list of predicted tags (achieved in test phase)
    :param ac_tag2ix: tag2index dictionary (made in the dataset pre-process vocabulary building phase)
    :return: fine-tuned tagging, list of spans for ac-tags (token indices)
    """
    ix2tag = {v:k for k,v in ac_tag2ix.items()}
    corected_preds = []
    loners = 0
    rebels = 0

    for seq in preds:
        corrected_seq = [ac_tag2ix["O"]] * len(seq)
        # in-span type counter
        type_dict = {type:0 for type in set([k.split("-")[-1] for k in ac_tag2ix.keys()]) if type != 'O'}

        i_beg = -1
        # handle first tag (change to B-x if stat with I-x)
        if seq[0] != ac_tag2ix['O']:
            i_beg = 0
            type_dict[ix2tag[seq[0]].split("-")[-1]] += 1

        for i in range(1,len(seq)):
            curr_tag = ix2tag[seq[i]]
            prev_tag = ix2tag[seq[i-1]]
            # prevent one-ord ACs (from training set - at least 3 words long and in genral must at least express opinion ("School Rocks)
            if (curr_tag != 'O') and (prev_tag == 'O') and (i!= len(seq)-1) and (ix2tag[seq[i+1]] == 'O'):
                # randomly set type according to immidiate neighbors
                curr_tag = 'O'
                seq[i] = ac_tag2ix['O']
                loners += 1

            # update counter for span type
            if curr_tag != 'O':
                type_dict[curr_tag.split("-")[-1]] += 1
                # if starting span (previous 'O' and current is a tag)
                if prev_tag =='O':
                    i_beg = i

            # if ending span (current 'O' and previous has a tag) or reaching end of sequence with non-'O' tag # TODO - consider randomize in cases of equality
            elif (curr_tag == 'O' and prev_tag != 'O') or (curr_tag != 'O' and i == len(seq)):
                max_type = max(type_dict, key = lambda k: type_dict[k])
                corrected_seq[i_beg] = ac_tag2ix["B-{}".format(max_type)]
                corrected_seq[i_beg+1:i] = [ac_tag2ix["I-{}".format(max_type)]] * (i - i_beg - 1)
                # reset type counter and span index
                for type in type_dict.keys(): type_dict[type] = 0
                rebels += (torch.tensor(seq[i_beg:i]) != torch.tensor(corrected_seq[i_beg:i])).sum().item()
                i_beg = -1

        # choose spans by majority
        corected_preds.append(corrected_seq)

    sys.stdout.write("post process completed.\nconverted {:d} incoherent lables and {:d} stand-alone ACs\n".format(rebels,loners))
    return corected_preds



def main(mode, config_file_path, trained_model_path):
    # train the segmentor-classifier first
    h_params = HyperParams(config_file_path)
    from src.preprocess import get_train_test_split
    from src.preprocess import prepare_data
    from src.models import BiLSTM_Segmentor_Classifier
    from src.models import BiLSTM_Segmentor_Classifier_no_pos

    torch.manual_seed(h_params.rand_seed)
    # debug - TODO delete - run for all models:
    model_dir = os.path.split(trained_model_path)[0]
    all_models = os.listdir(model_dir)
    for model_name in all_models:
        h_params.use_pos = True if model_name.find("no_POS") == -1 else False
        for mode in [DivisionResolution.ESSAY,DivisionResolution.SENTENCE,DivisionResolution.PARAGRAPH]:
            # Debug - TODO - delete after running all tests
            _, test_files = get_train_test_split(os.path.abspath(os.path.join("..", "data", "train-test-split.csv")))
            test_data, ept_offsets = prepare_data(mode, test_files)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            SegmentorClassifier = BiLSTM_Segmentor_Classifier if h_params.use_pos else BiLSTM_Segmentor_Classifier_no_pos
            model = SegmentorClassifier(h_params.d_word_embd, h_params.d_pos_embd, h_params.d_h1,
                                        h_params.n_lstm_layers, h_params.word_voc_size, h_params.pos_voc_size,
                                        h_params.ac_tagset_size, h_params.batch_size, device,
                                        h_params.pretraind_embd_layer_path)
            # load trained model state-dict
            checkpoint = torch.load(os.path.join(model_dir,model_name))  # TODO - delete when back to normal
            #checkpoint = torch.load(trained_model_path) #TODO - uncomment when back to normal
            model.load_state_dict(checkpoint['model_state_dict'])
            ## set CUDA if available
            if torch.cuda.is_available():
                model.cuda()
            # set evaluation mode mode
            model.eval()
            # inference for all chosen data
            preds = []
            with torch.no_grad():
                for (indexed_tokens, indexed_POSs, indexed_AC_tags) in test_data:
                    tag_scores = model((indexed_tokens.to(device),indexed_POSs.to(device))) # get log soft max for input
                    preds.append(torch.argmax(tag_scores, dim=1).tolist())
            # post-process for fine tuning
            ac_tag2ix = pickle.load(open(os.path.join("..","data","vocabularies","ac_tag2ix.pcl"),'rb'))
            corrected_tags = post_process(preds, ac_tag2ix)
            # save results
            #results_file = os.path.join("..","exps",os.path.split(trained_model_path)[-1][:-3]+".results") # TODO - uncomment when back to normal
            results_file = os.path.join("..","exps","{}|{}.results".format(model_name[:-3],mode)) # TODO - delete when back to normal

            true_tags = [ac_tags.tolist() for _,_,ac_tags in test_data]
            with open(results_file,'wt') as f:
                # write header for file
                f.write("\t".join(("# essay_paragraph_token_index","true AC-tag","predicted AC-tag","post processed AC tag"))+'\n')
                # iterate over results (by appropriate devision)
                for i_seq in range(len(preds)):
                    for i_tok in range(len(preds[i_seq])):
                        e_p_t_index = ept_offsets[i_seq][i_tok]
                        true_tag = true_tags[i_seq][i_tok]
                        predicted_ac_tag = preds[i_seq][i_tok]
                        post_processed_tag = corrected_tags[i_seq][i_tok]
                        f.write("\t".join((str(e_p_t_index),str(true_tag),str(predicted_ac_tag),str(post_processed_tag))))
                        f.write('\n')

            sys.stdout.write("finished predictions and saved to {}".format(os.path.abspath(results_file)))


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