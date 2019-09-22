#from __future__ import print_function
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
import sys
import cPickle as pickle 

def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

ref_file = sys.argv[1]
hyp_file = sys.argv[2]
print ref_file
print hyp_file

refs = pickle.load(open(ref_file, 'rb'))
tst_refs = refs[2]

test_caps = []
for va in tst_refs[2]:
    tmp = []
    for t in va:
        tmp2 = ""
        for tt in t:
            tmp2 = tmp2 + tt + " "
        tmp.append(tmp2[:-1].encode('ascii', 'ignore').decode('ascii'))
    test_caps.append(tmp)

test_refs = [tst_refs[1],test_caps]

refs = {idx: ref for (idx, ref) in enumerate(test_refs[1])}

# tst_caps = [[[" ".join(sents.strip()) for sents in story] for story in stories] for stories in tst_refs[2]]
# tst_caps = {idx: cap for (idx, cap) in enumerate(tst_caps)}
hyps = [line.strip() for line in open(hyp_file, 'r')]
hyps = {idx: [h] for (idx, h) in enumerate(hyps)}

scores = score(refs, hyps)
print scores



