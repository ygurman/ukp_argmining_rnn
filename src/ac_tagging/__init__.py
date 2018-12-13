import sys

import torch


def post_process(preds, ac_tag2ix):
    """
    inforce BIO restriction to improve classification (component begins with "B", type chosen by majority, only "I" within AC and no solitary ac's
    :param preds: list of predicted tags (achieved in test phase)
    :param ac_tag2ix: tag2index dictionary (made in the dataset pre-process vocabulary building phase)
    :return: fine-tuned tagging, list of spans for ac-tags (token indices)
    """
    ix2tag = {v:k for k,v in ac_tag2ix.items()}
    corected_preds = []
    loners = 0
    rebels = 0

    for seq in preds:
        corrected_seq = [ac_tag2ix["O"]] * len(seq)
        # in-span type counter
        type_dict = {type:0 for type in set([k.split("-")[-1] for k in ac_tag2ix.keys()]) if type != 'O'}

        i_beg = -1
        # handle first tag (change to B-x if stat with I-x)
        if seq[0] != ac_tag2ix['O']:
            i_beg = 0
            type_dict[ix2tag[seq[0]].split("-")[-1]] += 1

        for i in range(1,len(seq)):
            curr_tag = ix2tag[seq[i]]
            prev_tag = ix2tag[seq[i-1]]
            # prevent one-ord ACs (from training set - at least 3 words long and in genral must at least express opinion ("School Rocks)
            if (curr_tag != 'O') and (prev_tag == 'O') and (i!= len(seq)-1) and (ix2tag[seq[i+1]] == 'O'):
                # randomly set type according to immidiate neighbors
                curr_tag = 'O'
                seq[i] = ac_tag2ix['O']
                loners += 1

            # update counter for span type
            if curr_tag != 'O':
                type_dict[curr_tag.split("-")[-1]] += 1
                # if starting span (previous 'O' and current is a tag)
                if prev_tag =='O':
                    i_beg = i

            # if ending span (current 'O' and previous has a tag) or reaching end of sequence with non-'O' tag # TODO - consider randomize in cases of equality
            elif (curr_tag == 'O' and prev_tag != 'O') or (curr_tag != 'O' and i == len(seq)):
                max_type = max(type_dict, key = lambda k: type_dict[k])
                corrected_seq[i_beg] = ac_tag2ix["B-{}".format(max_type)]
                corrected_seq[i_beg+1:i] = [ac_tag2ix["I-{}".format(max_type)]] * (i - i_beg - 1)
                # reset type counter and span index
                for type in type_dict.keys(): type_dict[type] = 0
                rebels += (torch.tensor(seq[i_beg:i]) != torch.tensor(corrected_seq[i_beg:i])).sum().item()
                i_beg = -1

        # choose spans by majority
        corected_preds.append(corrected_seq)

    sys.stdout.write("post process completed.\nconverted {:d} incoherent lables and {:d} stand-alone ACs\n".format(rebels,loners))
    return corected_preds
