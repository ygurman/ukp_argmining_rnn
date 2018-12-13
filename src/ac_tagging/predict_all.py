# prediction using the segmentor-classifier for ACs
import pickle
import sys
from argparse import ArgumentParser
import os
import torch
from src.ac_tagging import post_process
from src.utils import HyperParams, DivisionResolution, mode_dict


def main(mode, config_file_path, trained_model_path):
    # train the segmentor-classifier first
    h_params = HyperParams(config_file_path)
    from src.utils.preprocess import get_train_test_split
    from src.utils.preprocess import prepare_data
    from src.ac_tagging.models import BiLSTM_Segmentor_Classifier
    from src.ac_tagging.models import BiLSTM_Segmentor_Classifier_no_pos

    torch.manual_seed(h_params.rand_seed)
    # debug - TODO delete - run for all models:
    model_dir = h_params.models_dir
    all_models = os.listdir(model_dir)
    for model_name in all_models:
        h_params.use_pos = True if model_name.find("no_POS") == -1 else False
        for mode in [DivisionResolution.ESSAY,DivisionResolution.SENTENCE,DivisionResolution.PARAGRAPH]:
            # Debug - TODO - delete after running all tests
            _, test_files = get_train_test_split(os.path.abspath(os.path.join(h_params.data_dir, "train-test-split.csv")))
            test_data, ept_offsets = prepare_data(mode, test_files,h_params.data_dir)
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
            ac_tag2ix = pickle.load(open(os.path.join(h_params.vocab_dir,"ac_tag2ix.pcl"),'rb'))
            corrected_tags = post_process(preds, ac_tag2ix)
            # save results
            #results_file = os.path.join("..","exps",os.path.split(trained_model_path)[-1][:-3]+".results") # TODO - uncomment when back to normal
            results_file = os.path.join(h_params.exps_dir,"{}|{}.results".format(model_name[:-3],mode)) # TODO - delete when back to normal

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