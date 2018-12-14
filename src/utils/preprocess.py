## preprocess the UKP Annotated Essays v2 DB
# read .ann files and convert to CONLL style token-based Tab-delimited format (essay and/or paragraph level)
# format: {INDEX}|{TOKEN}|{POS}|{AC-BIO}|{AC-IND}|{REL-TAG} where:
#     {POS} - a Stanford's CoreNLP POS tagger output
#     {AC-BIO} - Argument Component tag (standard B-I-O tags with Entity types of {Premise, Claim, MajorClaim})
#     {AC-IND} - Argument Component index
#     {REL-TAG} - Argument Relation tag of form "{R-TYPE}:#" (Type from {Support,Attack,For,Against}, # is the AC-IND of related AC)

### NLTK's Stanfords CoreNLP wrapper - https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK

import subprocess
from tqdm import tqdm
import os
import numpy as np
import pickle
import sys
from typing import Dict
import torch
import torch.nn as nn
import pydot
from nltk.parse import CoreNLPParser
from nltk.tokenize import sent_tokenize

EMPTY_SIGN = "~"

###
# generic .ann files related methods
###

def readAnnotatedFile(ann_path: str) -> (dict, dict, dict, list, list):
    """
    parse data from ".ann" UKP 2.0 files and pass according dictionaries for props, labels, stances, supports and attaks
    :param ann_path:
    :return:
    """
    propositions, prop_labels, prop_stances, supports, attacks = {}, {}, {}, [], []
    with open(file=ann_path, mode='rt', encoding='utf8') as f:
        for line in f:
            delimited = line.split('\t')
            typ = delimited[0][0]  # T == proposition , A = Stance, R = link
            inner_index = int(delimited[0][1:])
            data = delimited[1].split()

            if typ == 'T':
                label = data[0]  # prop lable (Premise, Cliam or MajorClaim)
                start, end = int(data[1]), int(data[2])  # proposition offsets
                propositions[inner_index] = (start, end)  # represent propositions by it's index boundries
                prop_labels[inner_index] = label

            elif typ == 'A':
                _, target_index, stance_value = data  # first Column in "A" lines is always "Stance", stance value in {For, Against}
                prop_stances[int(target_index[1:])] = stance_value

            elif typ == 'R':
                link_typ = data[0]  # link type in {supports, attacks}
                source, target = int(data[1][6:]), int(data[2][
                                                       6:])  # get inner indices of related propositions (ex:Arg1:T4 Arg2:T3 -> source == 4 , target = 3)
                link_list = supports if link_typ == 'supports' else attacks
                link_list.append((source, target))

    return propositions, prop_labels, prop_stances, supports, attacks

###
# visualize all argumentative essays and save to fixed location
###
def visualize_all_dataset(data_path = os.path.abspath(os.path.join("..","data"))):
    essays = [fn[:-4] for fn in os.listdir(data_path) if fn[:-3] == "ann"]
    for essay in essays:
        arg_doc = ArgDoc(os.path.join(data_path,essay))
        arg_doc.visualize(os.path.abspath(os.path.join("..","graphs")))
###
# ann 2 conll functions
###

# calculate new proposition offsets w/o spaces
def calc_no_spaces_indices(arg_doc:ArgDoc)->[(int,int)]:
    old_indices = arg_doc.prop_offsets
    text = arg_doc.text
    new_offsets = []
    for (beg,end) in old_indices:
        new_beg = len(text[:beg].replace(" ","").replace("\n",""))
        new_end = new_beg + len(text[beg:end].replace(" ","").replace("\n",""))
        new_offsets.append((new_beg, new_end))
    return new_offsets

# read .ann and .txt of essay# and craete appropriate token-level conll-like file as mentioned above
def pre_process_ukp_essay(base_path, pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')):
    arg_doc = ArgDoc(base_path)

    i_no_space = 0
    paragraphs = []
    # list of sentences (list of list of tuples representing tokens and POSs)
    for paragraph in arg_doc.text.split('\n'):
        # token,POS-tag,no_space_index tuple
        tagged_sentences = []
        # skip empty lines (usually seperated from essay title)
        if len(paragraph) == 0:
            continue
        # use nltk sentence tokenizer (PunktSentenceTokenizer)
        sentences = sent_tokenize(paragraph)
        for sent in sentences:
            # use Stanford's CoreNLP for POS tagging sentence by sentence
            pos_tagged_sent = pos_tagger.tag(sent.split())
            tok_pos_noSpaceIndex_sent = []
            for tok, pos in pos_tagged_sent:
                tok_pos_noSpaceIndex_sent.append((tok,pos,i_no_space))
                i_no_space += len(tok)
            tagged_sentences.append(tok_pos_noSpaceIndex_sent)
        paragraphs.append(tagged_sentences)

    # add appropriate AC tags by propositions
    no_space_prop_offsets = calc_no_spaces_indices(arg_doc)

    base_dir, base_filename = os.path.split(base_path)
    output_file = os.path.join(base_dir, "processed", base_filename + ".tsv")
    with open(output_file, 'wt', encoding='utf8') as f_out:
        for i_paragraph in range(len(paragraphs)):
            f_out.write("# paragraph {}\n".format(i_paragraph))
            for tagged_sentence in paragraphs[i_paragraph]:
                f_out.write("# sent\n")
                for tok,pos,i_no_space in tagged_sentence:
                    # handle AC tagging where propositions apply
                    # inefficient but written in haste for 0ne-time use ...
                    for i_prop in range(len(no_space_prop_offsets)):
                        # if the current token is in proposition i_prop
                        if (i_no_space >= no_space_prop_offsets[i_prop][0] and i_no_space < no_space_prop_offsets[i_prop][1]):
                            # tag AC information as required (beginning(B) or middle(I) of proposition + AC type)
                            ac_type = arg_doc.prop_labels[i_prop]
                            if i_no_space == no_space_prop_offsets[i_prop][0]:
                                ac_bio_tag = "B-" + ac_type
                            else:
                                ac_bio_tag = "I-" + ac_type
                            # tag relation information according to AC type ({AC index:supports\attacks} for premise, For/Against for Claim, empty tab for MajorClaim)
                            rel_tag = EMPTY_SIGN
                            if (ac_type == "Premise"):
                                # either it supports or attacks a claim
                                support_prems, supported = zip(*arg_doc.supports)
                                if i_prop in support_prems:
                                    rel_tag = "supports:{}".format(supported[support_prems.index(i_prop)])
                                else:
                                    attack_prems, attacked = zip(*arg_doc.attacks)
                                    rel_tag = "attacks:{}".format(attacked[attack_prems.index(i_prop)])
                            elif (ac_type == "Claim"):
                                # Claims only have For or Against relation type (they refer to the essay's major claims)
                                rel_tag = "{}:{}".format(arg_doc.prop_stances[i_prop],EMPTY_SIGN)

                            f_out.write("\t".join((tok,pos,ac_bio_tag,str(i_prop),rel_tag)))
                            f_out.write("\n")
                            break
                    else:
                        f_out.write("\t".join((tok,pos,"O",EMPTY_SIGN,EMPTY_SIGN)))
                        f_out.write("\n")


def convert_dataset_to_conll(data_path):
    """
    iterate over all annotated essays and preprocess as conll-style files
    :param data_path:
    :return:
    """
    subprocess.call(os.path.abspath(os.path.join("..","shell_scripts","start_core_nlp_server.sh"))) # set up CORE-NLP server
    pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
    essays_base_names = set([f_name.split(".")[0] for f_name in os.listdir(data_path) if f_name[:5]=="essay"])
    pbar = tqdm(essays_base_names)
    for base_name in pbar:
        pre_process_ukp_essay(base_path=os.path.join(data_path,base_name),pos_tagger=pos_tagger)

###
# vocabulary related methods
###

UNK_TOKEN_SYMBOL = "UNKNOWN_TOKEN"
PAD_SYMBOL = "PAD_SYM"
torch.manual_seed(361)
np.random.seed(361)

RELATIONS_TYPES = ["no-relation","a supports b","a attacks b","a For b","a Against b"]
AC_TYPES = ["Premise","Claim","MajorClaim"]

def build_vocabularies(data_path,train_file_names):
    """
    read files in order to create vocabularies of tags (AC types and rel Types) and inputs (words, POSs)
    :param data_path: base data directory
    :param train_file_names: list of essays in the train set (following the "essay{3d}.tsv" convention)
    :return: save the vocabularies for later use
    """
    word_voc = set()
    pos_voc = set()
    ac_tag_voc = set()
    ac_rel_voc = set()

    # iterate over the pre-processed conll-like data to build vocabulary
    for essay in train_file_names:
        essay_path = os.path.join(data_path,"processed",essay + ".tsv")
        with open(essay_path,"rt") as f:
            for line in f:
                if line[0] == "#":
                    continue
                tok, pos, ac_tag, _, rel_tag = line.strip().split('\t')
                word_voc.add(tok.lower())
                pos_voc.add(pos)
                ac_tag_voc.add(ac_tag)
                ac_rel_voc.add(rel_tag.split(":")[0])

    # build indexed dictionaries for the vocabularies
    word2ix = dict((w, i) for i,w in enumerate(word_voc))
    pos2ix = dict((w, i) for i, w in enumerate(pos_voc))
    ac_tag2ix = dict((w, i) for i, w in enumerate(ac_tag_voc))
    ac_type2ix = dict((w, i) for i, w in enumerate(AC_TYPES))
    rel2ix = dict((w, i) for i, w in enumerate(RELATIONS_TYPES))

    # save the vocabs to processed data folder for later use
    voc_dir = os.path.abspath(os.path.join("..","data","vocabularies"))
    if not os.path.exists(voc_dir):
        os.mkdir(voc_dir)
    for name, dic in (("word_2ix", word2ix), ("pos2ix", pos2ix), ("ac_tag2ix", ac_tag2ix),
                      ("rel2x",rel2ix),("type2ix",ac_type2ix)):
        pickle.dump(dic,open(os.path.join(data_path,"vocabularies",name + ".pcl"),"wb"))

    sys.stdout.write("wrote vocabularies to {}\n".format(os.path.abspath(voc_dir)))
    return word2ix, pos2ix, ac_tag2ix, ac_type2ix, rel2ix


# use the original train-test split supplied with the UKP dataset
def get_train_test_split(split_path) -> ([str] , [str]):
    """
    takes the semi-colon delimited original UKP dataset train-test cut and returns list of test and train filenames
    :param split_path:
    :return:
    """
    train_set = []
    test_set = []
    with open(split_path) as f:
        f.readline() # skip the first line
        for line in f:
            name, set = line.strip().replace("\"","").split(";")
            if set == "TRAIN":
                train_set.append(name)
            else:
                test_set.append(name)
    return train_set, test_set


def prepare_glove_embeddings(glove_embds_path, glove_txt_name):
    """
    create a vocabulary and numpy vectors from GloVe txt file and create and store in appropriate pickle file
    :param glove_embds_path:
    :return:
    """
    words = []
    ix = 0
    word2ix = dict()
    vectors = []

    sys.stdout.write("loading GloVe vectors\n")
    with open(os.path.join(glove_embds_path,glove_txt_name),'rt') as f:
        for line in f:
            split = line.split()
            word = split[0]
            embd = np.array([float(val) for val in split[1:]])

            words.append(word)
            word2ix[word] = ix
            ix += 1
            vectors.append(embd)

    sys.stdout.write("saving files to {}\n".format(glove_embds_path))
    pickle.dump(words,open(os.path.join(glove_embds_path,glove_txt_name.replace(".txt","_words_lst.pcl")),"wb"))
    pickle.dump(word2ix, open(os.path.join(glove_embds_path, glove_txt_name.replace(".txt", "_word2ix_dict.pcl")), "wb"))

    glove_embds = {w: vectors[word2ix[w]] for w in words}
    pickle.dump(glove_embds, open(os.path.join(glove_embds_path, glove_txt_name.replace(".txt", "_word2vectors_dict.pcl")), "wb"))

    sys.stdout.write("files saved to {}\n".format(glove_embds_path))

    return words, word2ix, glove_embds


def combine_word_vocab(dataset_voc:Dict[str,int], pretrained_embds:Dict[str,np.array], d_embd:int, save_path="."):
    """
    combine dataset's vocabulary with GloVe to create an embedding layer for words (initiating unkown tokens with random normal distributed weights)
    :param dataset_voc: a word2ix dictionary
    :param pretrained_embs_voc:
    :return: vocabulary_size, embeddings_dimentions, embedding layer with appropriate weights
    """
    new_word2ix = {w: i for (i,w) in enumerate(pretrained_embds.keys(),1)} #save index 0 for padding (if implementing batching
    new_word2ix[PAD_SYMBOL] = 0 # reserve un-initalized weight vector for padding symbol

    new_weights = dict()
    new_ix = len(new_word2ix.keys())

    diff = 0

    std = 1/np.sqrt(d_embd) # standard deviation for randomized vectors

    for ds_word in dataset_voc.keys():
        if ds_word not in pretrained_embds.keys():
            diff += 1
            # randomize normal distributed weight for unknown word using Xavier's initialization variant
            weight = np.random.normal(0, scale=std, size=(d_embd,)).astype(np.float32)
            new_word2ix[ds_word] = new_ix
            new_weights[new_ix] = weight
            new_ix += 1

    # add UNKNOWN_TOKEN_SYMBOL for later use
    new_word2ix[UNK_TOKEN_SYMBOL] = new_ix
    new_weights[new_ix] = np.random.normal(0, scale=std, size=(d_embd,))
    new_weights[new_word2ix[PAD_SYMBOL]] = np.zeros(d_embd,)

    # create weight matrix
    voc_size = len(new_word2ix.keys())

    weight_matrix = np.zeros((voc_size,d_embd))

    for word, ix in new_word2ix.items():
        try:
            weight_matrix[ix] = pretrained_embds[word]
        except KeyError:
            weight_matrix[ix] = new_weights[ix]

    # create embedding layer
    embd_layer = nn.Embedding(num_embeddings=voc_size,embedding_dim=d_embd)
    embd_layer.weight.data = torch.Tensor(weight_matrix)

    if save_path:
        with open(os.path.join(save_path,"combined_word_voc_word2ix.pcl"), "wb") as f:
            pickle.dump(obj=new_word2ix,file=f)
        with open(os.path.join(save_path,"combined_word_voc_embdLayer.pcl"), "wb") as f:
            pickle.dump(obj=embd_layer,file=f)
        sys.stdout.write("\n")

    sys.stdout.write("saved combined vocab and embedding layer to {}\n".format(save_path))

    return new_word2ix, embd_layer

def create_combined_vocab_and_pretrained_embeddings_layer(data_path):
    """
    create combined vocabularies and word-embeddings layer for pre-trained and train data
    """
    # get train-test split
    split_path = os.path.join(data_path, "train-test-split.csv")
    train_files, _ = get_train_test_split(split_path)
    # build and save vocabularies as word2index dictionaries
    word2ix, pos2ix, ac_tag2ix, rel_2ix = build_vocabularies(data_path=data_path, train_file_names=train_files)
    # prepare and save pre-trained GloVe word embeddings as word2np.array indices
    word_embd_dim = 100
    save_path = os.path.abspath(os.path.join(data_path, "vocabularies"))
    sys.stdout.write("processing pre-trained embeddings\n")
    _, _, glove_embds = prepare_glove_embeddings(glove_embds_path=os.path.join(data_path, "glove.6B"),
                                                 glove_txt_name="glove.6B.100d.txt")
    sys.stdout.write("building combined vocabulary and pretrained embedding layer\n")
    _, word_embd_layer = combine_word_vocab(dataset_voc=word2ix, pretrained_embds=glove_embds, d_embd=word_embd_dim,
                                            save_path=save_path)


# utility function for input preparation
def prepare_sequence(seq:[str], to_ix:dict) -> torch.tensor:
    """
    use pre-defined indexed vocabulary and return a tensor of integers appropriate to string input (list of tokens/POSs/tags)
    used as input for the segmentor/classifier (for the first embedding layers)
    :param seq: list of tokens/POSs/tags
    :param to_ix: dictionary incexing the tokens/POSs/tags vocabulary
    :return: matching torch.tensor of integers representing input lists
    """
    idxs = []
    for w in seq:
        if w in to_ix.keys():
            idxs.append(to_ix[w])
        else:
            idxs.append(to_ix[UNK_TOKEN_SYMBOL])
    return torch.tensor(idxs, dtype=torch.long) # convert to torch tensor type


def prepare_data(division_type, files, data_path):
    """
    prepare data for training in units according to devision type (sentence, paragraph or essay level)
    :param data_path: data directory path
    :param files: data file names list (train\test)
    :param division_type: enumarated value depicting devision level of data
    :return: sequences for training/testing and tagging
    """
    word2ix = pickle.load(open(os.path.join(data_path,"vocabularies","combined_word_voc_word2ix.pcl"),'rb'))
    pos2ix = pickle.load(open(os.path.join(data_path,"vocabularies", "pos2ix.pcl"), 'rb'))
    ac_tag2ix = pickle.load(open(os.path.join(data_path,"vocabularies", "ac_tag2ix.pcl"), 'rb'))

    # store as list of essays (as list of paragraphs (as list of sentences))) for later flattening by division type
    indexed_essays = []
    for essay in files:
        indexed_essay = []
        indexed_paragraph = []
        indexed_sent = []
        i_tok = 0
        i_par = 0
        i_ess = int(essay[-3:])
        with open(os.path.join(data_path,"processed",essay+".tsv"),'rt') as f:
            for line in f:
                # if starting new paragraph
                if line[:5] == "# par":
                    if len(indexed_sent) > 0:
                        indexed_paragraph.append(indexed_sent)
                        indexed_sent = []
                    if len(indexed_paragraph) > 0:
                        indexed_essay.append(indexed_paragraph)
                        indexed_paragraph = []
                    i_par = int(line.strip().split()[-1])
                # if starting new sentence
                elif line[:5] == "# sen":
                    if len(indexed_sent) > 0:
                        indexed_paragraph.append(indexed_sent)
                        indexed_sent = []
                else:
                    tok, pos, ac_tag, _, _ = line.strip().split('\t')
                    try:
                        ind_tok = word2ix[tok.lower()]
                    except KeyError:
                        ind_tok = word2ix[UNK_TOKEN_SYMBOL]
                    ind_pos = pos2ix[pos]
                    ind_tag = ac_tag2ix[ac_tag]
                    # save original (essay,paragraph,token) index
                    e_p_t_offset = (i_ess, i_par, i_tok)
                    # append as (token, pos, tag, offset) tuple
                    indexed_sent.append((ind_tok, ind_pos, ind_tag,e_p_t_offset))
                    i_tok += 1

        # handle end of file
        indexed_paragraph.append(indexed_sent)
        indexed_essay.append(indexed_paragraph)
        indexed_essays.append(indexed_essay)

    indexed_tokens = []
    indexed_POSs = []
    indexed_AC_tags = []
    e_p_t_offsets = []

    # wrote on the fly for one-time use ... maybe use pandas instead...
    if division_type.name == "ESSAY":
        for essay in indexed_essays:
            indexed_tokens.append(torch.tensor([tok for paragraph in essay for sent in paragraph for (tok,_,_,_) in sent],dtype=torch.long))
            indexed_POSs.append(torch.tensor([pos for paragraph in essay for sent in paragraph for (_,pos,_,_) in sent],dtype=torch.long))
            indexed_AC_tags.append(torch.tensor([tag for paragraph in essay for sent in paragraph for (_,_,tag,_) in sent],dtype=torch.long))
            e_p_t_offsets.append([e_p_t_index for paragraph in essay for sent in paragraph for (_,_,_,e_p_t_index) in sent])

    elif division_type.name == "PARAGRAPH":
            for essay in indexed_essays:
                for paragraph in essay:
                    indexed_tokens.append(torch.tensor([tok for sent in paragraph for tok,_,_,_ in sent],dtype=torch.long))
                    indexed_POSs.append(torch.tensor([pos for sent in paragraph for _,pos,_,_ in sent],dtype=torch.long))
                    indexed_AC_tags.append(torch.tensor([tag for sent in paragraph for _,_,tag,_ in sent],dtype=torch.long))
                    e_p_t_offsets.append([e_p_t_index for sent in paragraph for _,_,_,e_p_t_index in sent])

    elif division_type.name == "SENTENCE":
        for essay in indexed_essays:
            for paragraph in essay:
                for sent in paragraph:
                    indexed_tokens.append(torch.tensor([tok for tok, _, _,_ in sent],dtype=torch.long))
                    indexed_POSs.append(torch.tensor([pos for _,pos,_,_ in sent],dtype=torch.long))
                    indexed_AC_tags.append(torch.tensor([tag for _, _,tag,_ in sent],dtype=torch.long))
                    e_p_t_offsets.append([e_p_t_index for _,_,_,e_p_t_index in sent])

    return list(zip(indexed_tokens, indexed_POSs, indexed_AC_tags)), e_p_t_offsets

###
# preprocess main - enable pre-process tasks by flags
###
from argparse import ArgumentParser
def main(convert, build_voc, visualize, data_path):
    """
    pre-process dataset, enabling building vocabularies, visualization and conversion to conll format
    :param convert: convert data to conll?
    :param build_voc: build vocabularies and pre-trained embedding layer?
    :param visualize: create png graph visualization of the essays
    :param data_path: dataset derictory path
    :return:
    """
    if convert:
        convert_dataset_to_conll(data_path)
    if build_voc:
        create_combined_vocab_and_pretrained_embeddings_layer(data_path)
    if visualize:
        visualize_all_dataset(data_path)

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--convert2conll', action='store_true', help="convert dataset to conll form")
    arg_parser.add_argument('-b','--build_vocabs', action='store_true', help="build vocabularies using pre-trained embeddings")
    arg_parser.add_argument('-v', '--visualize_dataset', action='store_true', help="visualise all essays in dataset")
    arg_parser.add_argument('-d', '--data_path', action='store', default="/../data", help="data directory path")
    args = arg_parser.parse_args(sys.argv[1:])
    data_path = os.path.abspath(args.data_path)
    main(args.convert2conll, args.build_vocabs, args.visualize_dataset, data_path)
