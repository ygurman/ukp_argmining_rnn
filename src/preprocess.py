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
from typing import Dict, List
import torch
import torch.nn as nn
import pydot
from nltk.parse import CoreNLPParser
from nltk.tokenize import sent_tokenize

EMPTY_SIGN = "~"

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


class ArgDoc(object):
    """
    class that represents an argumentative essay (using UKP 2.0 annotations)
    """
    def __init__(self, base_path):
        self.ess_id = int(base_path[-3:])  # essay id according to UKP naming convention
        self._txt_path = base_path + ".txt"  # essay text file path
        self._ann_path = base_path + ".ann"  # UKP annotated file path
        # read document's text
        with open(file=self._txt_path, mode='rt', encoding='utf8') as f:
            self.text = f.read()

        # get essay's paragraph's indices (seperated with '\n')
        self.paragraph_offsets = self._getNewLineIndices(self.text)

        # read annotated data from file
        propositions, prop_labels, prop_stances, supports, attacks = readAnnotatedFile(self._ann_path)

        # update proposition offsets, labels, stances and link types
        inner_indices, self.prop_offsets = zip(
            *sorted(propositions.items(), key=lambda x: x[1]))  # use the beginning index of propositions for sort

        # paragraph alignmnt of propositions (ordered by proposition's offsets)
        self.prop_paragraphs = [np.searchsorted(self.paragraph_offsets, start) - 1 for start, _ in self.prop_offsets]

        # invert indices for key management
        new_indices = {k: v for v, k in enumerate(inner_indices)}
        n_props = len(self.prop_offsets)

        # update fields with new inverted indices
        self.prop_labels = [prop_labels[inner_indices[i]] for i in range(n_props)]
        self.prop_stances = {new_indices[k]: v for k, v in prop_stances.items()}
        self.supports = [(new_indices[src], new_indices[trg]) for src, trg in supports]
        self.attacks = [(new_indices[src], new_indices[trg]) for src, trg in attacks]
        self.links = self.supports + self.attacks

    def _getNewLineIndices(self, text: str) -> np.array:
        """
        utility function that returns essay text's paragraphs offsets
        """
        i = 0  # assume first char always opens paragraph
        paragraph_indices = []
        while i != -1:
            paragraph_indices.append(i)
            i = text.find('\n', i + 1)
        return np.array(paragraph_indices)

    def visualize(self, save_path=os.getcwd()):
        """
        plot the essay as graph using pydot and save as .png file
        """
        arg_graph = pydot.Dot(graph_type='digraph')

        maj_claims = [("! " + self.text[self.prop_offsets[i][0]:self.prop_offsets[i][1]]) for i in
                      range(len(self.prop_labels)) if
                      self.prop_labels[i] == 'MajorClaim']  # handle more than 1 major cliam for main node
        # add the major claims node
        head_node = pydot.Node('\n'.join(maj_claims), style='filled',
                               fillcolor='#eeccdd')
        arg_graph.add_node(head_node)

        nodes = {}
        # add the premise and claims nodes
        for i in range(len(self.prop_labels)):
            if self.prop_labels[i] == 'MajorClaim':
                continue
            text = self.text[self.prop_offsets[i][0]:self.prop_offsets[i][1]]
            start = 0
            label = []
            next_i = -1
            for i_c in range(1, len(text)):
                if i_c < next_i:
                    continue
                if i_c % 30 == 0:
                    next_i = text.find(" ", i_c) + 1
                    if next_i > 0:
                        label.append(text[start:next_i - 1])
                        start = next_i
                    else:
                        label.append(text[start:])
                        start = len(text)
            if (i_c > start):
                rest = " " + text[start:]
                if len(rest.split()) == 1:
                    label[-1] += rest
                else:
                    label.append(text[start:])

            nodes[i] = pydot.Node(i,
                                  label='\n'.join(label),
                                  style='filled',
                                  fillcolor='#ccbbdd' if self.prop_labels[i] == 'Claim' else '#aabbdd'
                                  )
            arg_graph.add_node(nodes[i])
        # add edges
        # add the stances (cliams-majorClaims) edges
        for i, val in self.prop_stances.items():
            tmp_edge = pydot.Edge(nodes[i], head_node,
                                  label=val,
                                  labelfontcolor='red' if val == "Against" else 'green',
                                  color='red' if val == "Against" else 'green'
                                  )
            arg_graph.add_edge(tmp_edge)

        # add the support/attacks edges
        for src, trg in self.supports:
            tmp_edge = pydot.Edge(nodes[src], nodes[trg])
            arg_graph.add_edge(tmp_edge)
        for src, trg in self.attacks:
            tmp_edge = pydot.Edge(nodes[src], nodes[trg],
                                  style='dotted',
                                  color='red'
                                  )
            arg_graph.add_edge(tmp_edge)
        # display and save
        path = os.path.join(save_path, "essay{:3d}.png".format(self.ess_id).replace(" ","0"))
        arg_graph.write_png(path)
        print("saved png to {}".format(path))

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


UNK_TOKEN_SYMBOL = "UNKNOWN_TOKEN"
PAD_SYMBOL = "PAD_SYM"
torch.manual_seed(361)
np.random.seed(361)

# utility function for input preparation
def prepare_sequence(seq:[str], to_ix:dict) -> torch.tensor:
    """
    use pre-defined indexed vocabulary and return a tensor of integers appropriate to string input (list of tokens/POSs/tags)
    used as input for the segmentor/classifier (for the first embedding layers)
    :param seq: list of tokens/POSs/tags
    :param to_ix: dictionary incexing the tokens/POSs/tags vocabulary
    :return: matching torch.tensor of integers representing input lists
    """
    idxs = [to_ix[w] for w in seq] # replace input tokens/POSs with matching indices
    return torch.tensor(idxs, dtype=torch.long) # convert to torch tensor type

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
                tok, pos, ac_tag, _, rel_tag = line.split('\t')
                word_voc.add(tok.lower())
                pos_voc.add(pos)
                ac_tag_voc.add(ac_tag)
                ac_rel_voc.add(rel_tag.split(":")[0])

    # build indexed dictionaries for the vocabularies
    word2ix = dict((w, i) for i,w in enumerate(word_voc))
    pos2ix = dict((w, i) for i, w in enumerate(pos_voc))
    ac_tag2ix = dict((w, i) for i, w in enumerate(ac_tag_voc))
    rel_2ix = dict((w, i) for i, w in enumerate(ac_rel_voc))

    # save the vocabs to processed data folder for later use
    voc_dir = os.path.abspath(os.path.join("..","data","vocabularies"))
    if not os.path.exists(voc_dir):
        os.mkdir(voc_dir)
    for name, dic in (("word_2ix", word2ix), ("pos2ix", pos2ix), ("ac_tag2ix", ac_tag2ix), ("ac_rel2ix", rel_2ix)):
        pickle.dump(dic,open(os.path.join(data_path,"vocabularies",name + ".pcl"),"wb"))

    sys.stdout.write("wrote vocabularies to {}\n".format(os.path.abspath(voc_dir)))
    return word2ix, pos2ix, ac_tag2ix, rel_2ix


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

def create_combined_vocab_and_pretrained_embeddings_layer():
    """
    create combined vocabularies and word-embeddings layer for pre-trained and train data
    """
    # get train-test split
    data_path = os.path.abspath(os.path.abspath(os.path.join("..", "data")))
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

def main():
    pass #TODO - apply preprocess by argparse choices

if __name__ == '__main__':
    main()