import os
import pickle
import sys
from collections import defaultdict

RELATIONS_TYPES = ["no-relation","a supports b","a attacks b","a For b","a Against b"]
def create_rel_vocabulary(relation_types,voc_dir_path):
    rel2ix = dict((w, i) for i, w in enumerate(relation_types))
    ix2rel = dict((i, w) for w, i in rel2ix.items())
    pickle.dump(rel2ix,open(os.path.join(voc_dir_path,"rel2ix.pcl"),'wb'))
    pickle.dump(ix2rel, open(os.path.join(voc_dir_path, "ix2rel.pcl"),'wb'))

class ArgComp(object):
    """
    represents argument component for pair-wise relation classification
    """
    def __init__(self,essay:int,paragraph:int,span:(int,int),AC_type:str,AC_id:int,tokens:[str],poss:[str]):
        self.essay = essay
        self.paragraph = paragraph
        self.span = span
        self.type = AC_type
        self.id = AC_id
        self.tokens = tokens
        self.poss = poss

    # comparison methods for structered features in relation classification
    def before(self,other):
        """
        indicates if this Argument component preceeds another argument component
        :param other: argument component
        """
        return self.span[0] < other.span[0]

    def distance(self,other):
        return abs(self.id - other.id)

    def same_paragraph(self,other):
        return self.paragraph == other.paragraph

def get_all_relations_from_tsv(essay_tsv_path,save_path=None):
    """
    :param essay_tsv_path: pre-processed Conll style tsv file
    :return: dictionary of all ACs (key=ac_id), dictionary of all relations (key=tuple of ids) - non related ACs pairs included for negative examples
    """
    ac_dict = {}
    major_claims = set()
    relations_dict = dict()

    # get all ACs
    # gather paragraph offsets and major claims ids
    with open(essay_tsv_path,'rt') as f:
        lines = f.readlines()
        tokens_buffer = []
        poss_buffer = []
        par = 0
        span_begin = 0
        prev_role = 'O'
        prev_type = 'O'
        prev_id = 0
        offsets = []
        ac_type = 'O'

        # read al ACs from tsv preprocessed CONLL-style file
        for i_line in range(len(lines)):
            line = lines[i_line]
            if line[:5] == "# par":
                par = line.split()[-1]
            elif line[0] != "#":
                tok, pos, ac_tag, ac_id, _ = line.split()
                role = ac_tag[0]
                # handle ac beginning
                if role == "B":
                    span_begin = i_line
                    offsets.append(i_line+1)

                # handle either in-component or beginning
                if role != 'O':
                    ac_type = ac_tag.split("-")[1]
                    tokens_buffer.append(tok)
                    poss_buffer.append(pos)

                    if ac_type == "MajorClaim":
                        # save major calims ids for relation specification
                        major_claims.add(int(ac_id))

                # handle end of ac
                if (role == 'O' and prev_role != 'O') or (role != 'O' and i_line == len(lines)) :
                    # create ac
                    ac = ArgComp(essay=int(essay_tsv_path[-7:-4]),paragraph=int(par),span=(span_begin,i_line),
                                 AC_type=prev_type,AC_id=int(prev_id),tokens=tokens_buffer,poss=poss_buffer)
                    # add ac to dictionary
                    ac_dict[ac.id] = ac
                    # reset values (empty buffers)
                    tokens_buffer = []
                    poss_buffer = []

                prev_role = role
                prev_id = ac_id
                prev_type = ac_type

    # get all relations (iterate over collected offsets (spans beginnings)
    # create double entries for 2-way interpertation
    for i in offsets:
        _,_,_,a_id,rel = lines[i].split()
        # it's a MajorClaim - has no relations (root of argument tree)
        if rel == '~':
            continue
        else:
            rel_type,b_id = rel.split(":")
            # if it's a claim - add relations to major claims
            if b_id == "~":
                for mc in major_claims:
                    relations_dict[(int(a_id),mc)] = "a {} b".format(rel_type)
                    relations_dict[(mc,int(a_id))] = "b {} a".format(rel_type)
            # if it's a support/attak just
            else:
                a_id = int(a_id)
                b_id = int(b_id)
                relations_dict[(a_id, b_id)] = "a {} b".format(rel_type)
                relations_dict[(b_id, a_id)] = "b {} a".format(rel_type)

    # add all no-relations to the relations dictionaries
    for a_id in ac_dict.keys():
        for b_id in ac_dict.keys():
            # skip self relation
            if a_id == b_id:
                continue
            relations_dict.setdefault((a_id,b_id),"no-relation")
            relations_dict.setdefault((b_id, a_id), "no-relation")

    if save_path:
        essay = os.path.split(essay_tsv_path)[-1][:-4]
        pickle.dump(relations_dict,open(os.path.join(save_path,"{}_relations_dict.pcl".format(essay)),'wt'))
        pickle.dump(ac_dict, open(os.path.join(save_path, "{}_ac_dict.pcl".format(essay)), 'wt'))

    return ac_dict, relations_dict

def convert_results_to_tsv(results_file_path, vocab_dir_path,data_path):
    """
    take the segmentor-classifier's predictions and convert to the CONLL-style tsv format (used in test)
    :param ix2actag_path: path to pickeled ix2tag dictionary
    :param results_file_path: path to predicted
    :return: create tsv files per essay for predicted ACs
    """
    # predicted form - essay_paragraph_token_index	true AC-tag	predicted AC-tag	post processed AC tag
    # conll form - token,pos,ac_tag(predicted),ac_id(need to calculate),gold_relation(calculate based on majority)
    ix2tag = pickle.load(open(os.path.join(vocab_dir_path,"ix2rel.pcl")))

    # first determine relevant essays from predicted file
    essays = set()
    predicted_ac_tags = defaultdict(list)
    with open(results_file_path,'rt') as f:
        next(f) # skip header
        for line in f:
            ess_id, _, _, predicted_tag = line.split('\t')
            essays.add(int(ess_id))
            predicted_ac_tags[int(ess_id)].append(ix2tag[int(predicted_tag)])

    # read from tsv file (tokens, poss, indices and relations as is)
    for essay in essays:
        paragraph_offsets = []
        # copy info from tsv
        with open(os.path.join(data_path,"processed","essay{:3d}.tsv".format(essay).replace(" ","0"))) as f_tsv:
            i_tok = 0
            new_ac_tags = predicted_ac_tags[essay]
            tokens = []
            poss = []
            old_ids = []
            old_rels = []
            new_spans = []
            span_beg = 0
            for line in f_tsv:
                if line[:5] == "# par":
                    paragraph_offsets.append(i_tok)
                elif line[0] != "#":
                    tok, pos, _, old_id, old_rel = line.split()
                    tokens.append(tok)
                    poss.append(pos)
                    old_ids.append(old_id)
                    old_rels.append(old_rel)
                    # if detecting new component start span
                    if (new_ac_tags[i_tok][0]=='B'):
                        span_beg = i_tok
                    # if ending span
                    elif (new_ac_tags[i_tok][0] == 'O' and i_tok > 0 and new_ac_tags[i_tok-1][0] != 'O'):
                        new_spans.append((span_beg,i_tok))
                    i_tok += 1

        # apply voting mechanism to fill empty gaps (use majority where there are disagreements about rel_type. use new ac_id
        old_ac_dict, old_rel_dict = get_all_relations_from_tsv(os.path.join(data_path,"processed","essay{:3d}.tsv".format(essay).replace(" ","0")))

        # first vote if matching to any old id (by majority) - note: this is invariant to wrong classification and only refer to border detection
        new_ids = [~] * len(new_ac_tags)
        map_new2old = defaultdict()
        new_rels = ["~"] * len(new_ac_tags)
        for new_id, span in list[enumerate(new_spans)]:
            beg, end = span
            # update new ac_ids
            new_ids[beg:end] = [new_id] * (end - beg)
            old_id_count = {id:old_ids[beg:end].count(id) for id in old_ids[beg:end]}
            # tie breaker favors ACs over no-AC
            if "~" in old_id_count.keys(): old_id_count["~"] -= 1
            majority_old_id = max(old_id_count, key = old_id_count.get)
            map_new2old[new_id] = majority_old_id
            rel_maj = old_rels[beg:end][old_ids.index(majority_old_id)]
            # storing old rels according to old ids in new rels (will be replaced using dictionary later)
            new_rels[beg:end] = [rel_maj] * (end-beg)

        map_old2new =  {old: new for new,old in map_new2old.items()}

        # compute new relations
        for span in new_spans:
            beg, end = span
            old_rel = new_rels[beg]
            # if no relation drop it
            if old_rel == "~":
                continue
            # either in form of {A/F}:{~} or {a/s}:{old_id}
            else:
                type, side_b = old_rel.split(":")
                # if it's the former, leave it
                if side_b != "~":
                    side_b = map_old2new[side_b]
                    new_rels[beg:end] = ["{}:{}".format(type,side_b)] * (end-beg)

        # create new tsv file
        dest_dir = os.path.split(results_file_path)[0]
        with open(os.path.join(dest_dir,"predicted_essay{:3d}.tsv".format(essay).replace(" ","0")),"wt") as new_tsv:
            for i in range(len(new_ac_tags)):
                new_id = new_ids[i]
                new_tsv.write("{}\t{}\t{}\t{}\t{}\n".format(tokens[i],poss[i],new_ac_tags[i],new_id,new_rels[i]))

        sys.stdout.write("wrote {} tsvs to {}".format(len(essay),dest_dir))

def prepare_data():
    pass

get_all_relations_from_tsv('/home/yochay/ukp_argmining_rnn/data/processed/essay012.tsv')