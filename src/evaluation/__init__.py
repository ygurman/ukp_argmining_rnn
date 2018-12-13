import os
import pickle
import sys

from sklearn.metrics import precision_recall_fscore_support

import pandas as pd
import numpy as np

def evaluate_segmentation_results(result_file_path,ix2tag_path,eval_path):
    """
    evaluate and display f1,recall,persicion and accuracy with and without segmentation
    :param result_file_path:
    :return: print results
    """
    tag2ix = pickle.load(open(ix2tag_path, 'rb'))
    data = pd.read_csv(result_file_path,delimiter='\t',header=0)
    y_true = data[['true AC-tag']].values
    y_pred = data[['predicted AC-tag']].values
    y_pred_processed = data[['post processed AC tag']].values

    # display results without post processing (use micro averaging since this is only to evaluate the different models roughly
    pred_persicion, pred_recall, pred_f1, _ = precision_recall_fscore_support(y_true=y_true,y_pred=y_pred,average='micro')
    pred_accuracy = (1.*np.sum(y_true == y_pred)) / len(y_true)

    # display results with post processing
    post_persicion, post_recall, post_f1, _ = precision_recall_fscore_support(y_true=y_true,y_pred=y_pred_processed,average='micro')
    post_accuracy = (1*np.sum(y_true == y_pred_processed)) / len(y_true)

    # show only border matching results
    y_true_only_borderd = np.array(y_true != tag2ix["O"])
    y_pred_only_borders = np.array(y_pred != tag2ix["O"])
    y_pred_processed_only_borders = np.array(y_pred_processed != tag2ix["O"])

    # w/o post processing
    border_pred_persicion, border_pred_recall, border_pred_f1, _ = precision_recall_fscore_support(y_true=y_true_only_borderd,y_pred=y_pred_only_borders, average='micro')
    border_pred_accuracy = (1*np.sum(y_true_only_borderd == y_pred_only_borders)) / len(y_true)

    # w post processing
    border_post_persicion, border_post_recall, border_post_f1, _ = precision_recall_fscore_support(
        y_true=y_true_only_borderd, y_pred=y_pred_processed_only_borders, average='micro')
    border_post_accuracy = (1.*np.sum(y_true_only_borderd == y_pred_only_borders)) / len(y_pred)

    model_name = os.path.split(result_file_path)[-1]

    with(open(eval_path,"a")) as f:
        f.write("\t".join((model_name, "no_post_processing","full",str(pred_persicion),str(pred_recall),str(pred_f1),str(pred_accuracy)))+"\n")
        f.write("\t".join((model_name,"post_processing","full",str(post_persicion),str(post_recall),str(post_f1),str(post_accuracy)))+"\n")
        f.write("\t".join((model_name, "no_post_processing","borders_only",str(border_pred_persicion),str(border_pred_recall),str(border_pred_f1),str(border_pred_accuracy)))+"\n")
        f.write("\t".join((model_name, "post_processing","borders_only",str(border_post_persicion),str(border_post_recall),str(border_post_f1),str(border_post_accuracy)))+"\n")

    sys.stdout.write('saved metrics to {}\n'.format(eval_path))

def build_conf_matrix():
    pass

def create_mistakes_dict():
    pass

def get_rough_summary():
    results_dir = '/home/yochay/ukp_argmining_rnn/exps'
    ix2tag_path = '/home/yochay/ukp_argmining_rnn/data/vocabularies/ac_tag2ix.pcl'
    eval_path = '/home/yochay/ukp_argmining_rnn/eval/segmentor_comparison.tsv'
    for result_file in os.listdir(results_dir):
        if result_file.split(".")[-1] != "results":
            continue
        evaluate_segmentation_results(os.path.join(results_dir,result_file),ix2tag_path,eval_path)

get_rough_summary()