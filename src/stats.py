# check how many paragraphs, sentences and essays in train and data splits
import os
import sys

from src.preprocess import get_train_test_split

data_path = os.path.join("..","data")
train_files, test_files = get_train_test_split(os.path.join(data_path,'train-test-split.csv'))
for ds in (train_files,test_files):
    n_tokens = 0
    n_sents = 0
    n_paragraphs = 0
    n_essays = 0
    len_para = 0
    for essay in ds:
        n_essays += 1
        i_line = 0
        with open(os.path.join(data_path,"processed",essay+".tsv")) as f:
            for line in f:
                i_line += 1
                if line[:3] == "# p":
                    if len_para == 0:
                        print("ess:{}\tline:{}".format(essay,i_line))
                    n_paragraphs += 1
                    len_para = 0
                elif line[:3] == "# s":
                    n_sents += 1
                else:
                    n_tokens += 1
                    len_para += 1
    sys.stdout.write("dataset:      \n"
                     "n_essays:     {}\n"
                     "n_paragraphs: {}\n"
                     "n_sentences:  {}\n"
                     "n_tokens:     {}\n\n".format(n_essays,n_paragraphs,n_sents,n_tokens))