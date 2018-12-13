# check how many paragraphs, sentences and essays in train and data splits
import os
import sys

from src.utils.preprocess import get_train_test_split

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

import os
from src.utils.preprocess import get_train_test_split


train, test = get_train_test_split(os.path.join("..","data","train-test-split.csv"))

test_shortest_ac_len = 100000
train_shortest_ac_len = 100000

for file in train:
    with open(os.path.join("..","data","processed",file + '.tsv')) as f:
        tmp = 0
        prev = 'O'
        for line in f:
            if line[0] == "#":
                continue
            if line.split()[2] != 'O':
                tmp += 1
            else:
                if prev != 'O' and tmp < train_shortest_ac_len:
                    train_shortest_ac_len = tmp
                    tmp = 0
            prev = line.split()[2]

for file in test:
    with open(os.path.join("..","data","processed",file + '.tsv')) as f:
        tmp = 0
        prev = 'O'
        for line in f:
            if line[0] == "#":
                continue
            if line.split()[2] != 'O':
                tmp += 1
            else:
                if prev != 'O' and tmp < test_shortest_ac_len:
                    test_shortest_ac_len = tmp
                    tmp = 0
            prev = line.split()[2]

print("shortest ACs len\ntrain:{}\ttest:{}".format(train_shortest_ac_len,test_shortest_ac_len))