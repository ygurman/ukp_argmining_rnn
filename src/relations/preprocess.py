RELATIONS_TYPES = ["no-relation","a supports b","a attacks b","a For b","a Against b"]

class ArgComp(object):
    """
    represents argument component for pair-wise relation classification
    """
    def __init__(self,essay,paragraph,span,AC_type,AC_id,tokens,poss):
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

def get_all_relations_from_gold(essay_tsv_path):
    """
    :param essay_tsv_path: pre-processed Conll style tsv file
    :return: list of all ACs, list of all relations
    """


    ac_dict = {}
    major_claims = set()
    relations = {"supports": [], "attacks": [], "for": [], "against": []}

    # get all ACs
    # gather paragraph offsets and major claims ids
    with open(essay_tsv_path,'rt') as f:
        lines = f.read().split('\n')
        tokens_buffer = []
        poss_buffer = []
        par = 0
        span_begin = 0
        i_tok = 0
        prev_role = ''
        ac_type = ''
        for line in lines:
            if line[:5] == "# par":
                par = line.split()[-1]
            elif line[0] != "#":
                tok, pos, ac_tag, ac_id, _ = line.split()
                role , ac_type = ac_tag.split("-")
                # handle ac beginning
                if role == "B":
                    span_begin =  i_tok

                # handle either in-component or beginning
                if role != 'O':
                    tokens_buffer.append(tok)
                    poss_buffer.append(pos)

                if ac_type == "MajorClaim":
                    # save major calims ids for relation specification
                    major_claims.add(line.split()[3])
                i_tok +=1

                prev_role = ac_tag.split("-")[0]

    # get all ACs


    # get all relations
    return ac_dict, relations

def get_all_relations_from_predicted(results_file_path):
    pass # TODO - create data for segmentor classifier results