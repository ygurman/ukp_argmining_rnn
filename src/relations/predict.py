# prediction using the segmentor-classifier for ACs
import os
import sys
from argparse import ArgumentParser

import torch

from src.models import BlandRelationClassifier, BiLSTMRelationClassifier, BaselineRelationClassifier, \
    BaselineConstructedRelationClassifier
from src.utils import HyperParams, prepare_relations_data
from src.utils.preprocess import get_train_test_split


def main(config_file_path, trained_model_path, use_gold_segmentation):
    # get hyper parameters
    h_params = HyperParams(config_file_path)

    torch.manual_seed(h_params.rand_seed)
    _, test_files = get_train_test_split(os.path.abspath(os.path.join("..", "data", "train-test-split.csv")))
    test_data = prepare_relations_data(files=test_files,data_dir=os.path.join(h_params.exps_dir,"best_results"),vocab_dir=h_params.vocab_dir,save=True)
    gold_data = prepare_relations_data(files=test_files, data_dir=os.path.join(h_params.data_dir, "processed"),vocab_dir=h_params.vocab_dir, save=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if trained_model_path.find("BiLSTMRelationClassifier_transfer") != 1:
        RelationsClassifier = BiLSTMRelationClassifier
    elif trained_model_path.find("BlandRelationClassifier") != 1:
        RelationsClassifier = BlandRelationClassifier
    elif trained_model_path.find("BaselineConstructedRelationClassifier") != 1:
        RelationsClassifier = BaselineConstructedRelationClassifier
    else:
        RelationsClassifier = BaselineRelationClassifier

    model = RelationsClassifier(h_params.d_word_embd, h_params.d_pos_embd, h_params.d_h1,h_params.n_lstm_layers,
                               h_params.word_voc_size, h_params.pos_voc_size,h_params.ac_tagset_size,
                               h_params.batch_size, device, h_params.pretraind_embd_layer_path,
                               h_params.rel_tagset_size,h_params.d_tag_embd,h_params.d_small_embd,
                               h_params.d_distance_embd, h_params.d_h2, h_params.d_h3)

    # load trained model state-dict
    checkpoint = torch.load(trained_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    ## set CUDA if available
    if torch.cuda.is_available():
        model.cuda()

    # set evaluation mode mode
    model.eval()

    # inference for all chosen data
    preds = []
    essay_pairs_types = []
    true_rel_tag = []
    with torch.no_grad():
        for (ac_dict, ac_pairs, rel_tags) in test_data:
            try:
                a_id, b_id = ac_pairs[0], ac_pairs[1]
                ac_a , ac_b = ac_dict[a_id], ac_dict[b_id]
                tag_scores = model((ac_a,ac_b)) # get log soft max for input
                preds.append(torch.argmax(tag_scores, dim=1).tolist())
                essay_pairs_types.append((ac_a.essay,ac_pairs,ac_a.type,ac_b.type))
                true_rel_tag.append(rel_tags[(ac_a,ac_b)])
            except:
                pass # for debug - bad preprocessed files

    # save results
    results_file = os.path.join(h_params.exps_dir,os.path.split(trained_model_path)[-1][:-3]+".results")

    with open(results_file,'wt') as f:
        # write header for file
        f.write("\t".join(("#essay","ac_id_pairs","ac_a type","ac_b type","prediction","y_true"))+'\n')
        # iterate over results (by appropriate devision)
        for i_pred in range(len(preds)):
            essay, pair, a_type, b_type = essay_pairs_types[i_pred]
            pred = preds[i_pred]
            true = true_rel_tag[i_pred]
            f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(essay,pair,a_type,b_type,pred,true))

    sys.stdout.write("finished predictions and saved to {}".format(os.path.abspath(results_file)))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-cp', '--config_path', default=os.path.abspath(os.path.join("..", "params.conf")),
                        help=" path to learning parameters file")

    parser.add_argument('-mp', '--model_path', required=True, help=" path to trained model")

    parser.add_argument('-g', '--gold_segmentation', action='store_true',
                        help="use gold segmentation")

    args = parser.parse_args(sys.argv[1:])
    main(os.path.abspath(args.config_path), os.path.abspath(args.model_path), args.gold_segmentation)