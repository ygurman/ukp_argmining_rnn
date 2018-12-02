## preprocess the UKP Annotated Essays v2 DB
# read .ann files and convert to CONLL style token-based Tab-delimited format (essay and/or paragraph level)
# format: {INDEX}|{TOKEN}|{POS}|{AC-BIO}|{AC-IND}|{REL-TAG} where:
#     {POS} - a Stanford's CoreNLP POS tagger output
#     {AC-BIO} - Argument Component tag (standard B-I-O tags with Entity types of {Premise, Claim, MajorClaim})
#     {AC-IND} - Argument Component index
#     {REL-TAG} - Argument Relation tag of form "{R-TYPE}:#" (Type from {Support,Attack,For,Against}, # is the AC-IND of related AC)

### NLTK's Stanfords CoreNLP wrapper - https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK

import os
import numpy as np


# parse data from ".ann" UKP 2.0 files
def readAnnotatedFile(ann_path: str) -> (dict, dict, dict, list, list):
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


def getNewLineIndices(text: str) -> np.array:
    i = 0  # assume first char always opens paragraph
    paragraph_indices = []
    while i != -1:
        paragraph_indices.append(i)
        i = text.find('\n', i + 1)

    return np.array(paragraph_indices)


class ArgDoc(object):
    def __init__(self, base_path):
        self.ess_id = int(base_path[-3:])  # essay id according to UKP naming convention
        self._txt_path = base_path + ".txt"  # essay text file path
        self._ann_path = base_path + ".ann"  # UKP annotated file path
        # read document's text
        with open(file=self._txt_path, mode='rt', encoding='utf8') as f:
            self.text = f.read()

        # get essay's paragraph's indices (seperated with '\n')
        self.paragraph_offsets = getNewLineIndices(self.text)

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

# run the "start_core_nlp_server.sh" script before (sets a stanfoed's corenlp server in port 9000)
from nltk.parse import CoreNLPParser
from nltk.tokenize import sent_tokenize

EMPTY_SIGN = "~"

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
                    # inefficient but written in haste for 0ne-time use ... TODO: improve later
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


from tqdm import tqdm

def main():
    # iterate over all annotated essays and preprocess as conll-style files
    pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
    data_path = r'/home/yochay/arg_mining_proj/data'
    essays_base_names = set([f_name.split(".")[0] for f_name in os.listdir(data_path) if f_name[:5]=="essay"])
    pbar = tqdm(essays_base_names)
    for base_name in pbar:
        pre_process_ukp_essay(base_path=os.path.join(data_path,base_name),pos_tagger=pos_tagger)

if __name__ == '__main__':
    main()

