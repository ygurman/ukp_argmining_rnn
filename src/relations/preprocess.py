import os
import pickle

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

def get_all_relations_from_predicted(results_file_path):
    pass # TODO - create data for segmentor classifier results


def prepare_data():
    pass

get_all_relations_from_tsv('/home/yochay/ukp_argmining_rnn/data/processed/essay012.tsv')