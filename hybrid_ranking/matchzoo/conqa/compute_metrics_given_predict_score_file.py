'''
The script for computing the final bleu/rouge metrics of the hybrid
neural re-ranker of the hybrid neural conversational model

Given the prediction score file of matchzoo, compute the bleu/rouge metrics
of the hybrid retrieval-generation hybrid neural conversation model with
rescoring the mixture of the generated/retrieved response candidates

Updated on 08062018. Added evaluation metrics for METEOR and CIDEr
Hidden METEOR and CIDEr since the computation time cost for these two metrics is too high

Update on 08152018. Add distinct-1 and distinct-2 to evaluate the diversity
of n-grams in the generated responses
'''

import sys
import os
sys.path.append('../../metrics/')
import numpy as np
import nltk

reload(sys)
sys.setdefaultencoding('utf8')

from eval_test import score_given_a_sequence_pair, compute_distinct_n_gram
from tqdm import tqdm

def read_refs(reference_file):
    return [post_process(l) for l in open(reference_file).readlines()]

def read_hyps(hypotheses_file):
    hyps = {} # id to hypos
    with open(hypotheses_file) as fin:
        index = -1
        for l in fin:
            index += 1
            hyps[index] = l.strip()
        return hyps

def read_gen_resp(hypotheses_file):
    hyps = {} # test_id to hypos
    with open(hypotheses_file) as fin:
        index = -1
        for l in fin:
            index += 1
            hyps['test_' + str(index)] = l.strip()
        return hyps

# here we do post-processing for _ _ mention _ _ and _ _ url _ _ to avoid bais on the computation of BLUE score
# transfer &apos; to '
def post_process(ori_response):
    processed_response = ori_response.replace('_ _ mention _ _', 'mentionplaceholder',).replace('_ _ url _ _', 'urlplaceholder')
    processed_response = processed_response.replace('&apos;', "'")
    return processed_response

def init_test_id_to_gen_seq(mz_score_file, corpus_dict):
    # test_607        Q0      35313   0       1.289269        MatchPyramid    1
    # test_607        Q0      450886  1       0.841863        MatchPyramid    0
    # test_607        Q0      626134  2       0.832523        MatchPyramid    0
    test_id_to_gen_seq = {}
    test_id_selected_response_id = {}
    with open(mz_score_file) as fin:
        for l in fin:
            tok = l.split('\t')
            #print('test tok: ', tok)
            if tok[3] == '0': # only store the top-1 response in the rank list
                test_id_to_gen_seq[tok[0]] = corpus_dict[tok[2]]
                test_id_selected_response_id[tok[0]] = tok[2]
    print('len of test_id_to_gen_seq :', len(test_id_to_gen_seq))
    #print('test_id_to_gen_seq :', test_id_to_gen_seq)
    return test_id_to_gen_seq, test_id_selected_response_id

def compute_bleu_rouge_given_scores_in_train(mz_score_list, corpus_dict, reference_list, tag_partition):
    '''
    Compute BLEU/ROUGE-L metrics in match-zoo during the training process of the neural re-ranker
    :param mz_score_list: the list to store the predicted mz matching scores
    :param corpus_dict: the dict to store the mapping from text_id to text
    :param reference_list: the list to store the ground truth response (should be post-processed)
    :param tag_partition: a tag to indicate this is valid or test data
    :return: a result string which stores the current BLEU/ROUGE-L related metrics
    '''
    test_id_to_gen_seq = {}
    for score_l in mz_score_list:
        # score_l format: qid Q0 docid rank score method_label groud_truth_label
        tok = score_l.split('\t')
        if tok[3] == '0':
            test_id_to_gen_seq[tok[0]] = corpus_dict[tok[2]]
    # print('len of test_id_to_gen_seq :', len(test_id_to_gen_seq))
    # print('test_id_to_gen_seq:', test_id_to_gen_seq)
    metrics_keys = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', ]  # 'METEOR'
    metrics = np.zeros((len(reference_list), len(metrics_keys)))
    test_id_to_metrics_dict = {}
    index = -1
    gen_response_avg_len = 0.0
    gen_response_num = 0.0
    list_of_references = []
    hypotheses = []
    gen_response_words_corpus = []
    partition = tag_partition if tag_partition == 'test' else 'dev'
    for ref_l in reference_list:
        index += 1
        gt_response = ref_l.strip()
        qid = partition + '_' + str(index)
        if qid in test_id_to_gen_seq:
            gen_response = test_id_to_gen_seq[partition + '_' + str(index)].strip()
        else:
            gen_response = "null" # if qid is not in test_id_to_gen_seq, this means there are no retrieved responses (the generated response has been removed in only_ret_rerank method)
        gen_response_words = gen_response.lower().split(' ') # already post processed
        gt_response_words = gt_response.lower().split(' ') # already post processed
        gen_response_words_corpus.extend(gen_response_words)
        gen_response_avg_len += len(gen_response_words)
        hypotheses.append(gen_response_words)
        list_of_references.append([gt_response_words])
        gen_response_num += 1.0
        test_id_to_metrics_dict[index] = score_given_a_sequence_pair(gt_response, gen_response)
        for m_i in range(0, len(metrics_keys)):
            metrics[index, m_i] = test_id_to_metrics_dict[index][metrics_keys[m_i]]
    avg_metrics = [i * 100 for i in np.mean(metrics, axis=0).tolist()]
    avg_len = gen_response_avg_len / gen_response_num
    # print('(BLEU-1 BLEU-2 BLEU-3 BLEU-4, ROUGE-L) ', avg_metrics) # METEOR
    # print('average length of generated responses: ', avg_len)
    BLEUscore_Corpus = nltk.translate.bleu_score.corpus_bleu(list_of_references, hypotheses) * 100
    dist1 = compute_distinct_n_gram(gen_response_words_corpus, 1)
    dist2 = compute_distinct_n_gram(gen_response_words_corpus, 2)
    #print ('CorpusBLEU ', BLEUscore_Corpus)
    res_string = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (
                avg_metrics[0], avg_metrics[1], avg_metrics[2] \
                , avg_metrics[3], BLEUscore_Corpus, avg_metrics[4], dist1, dist2, avg_len)
    return res_string

def compute_bleu_rouge_given_single_file(mz_score_file, corpus_dict, is_mz_score_file,
                                         example_file):
    ''' compute blue and rouge scores given a score file from MatchZoo
    generated the exmaple file. Add the response id information to identify
    whether the response is from retrieved results or generated results in
    the hybrid models
    :param mz_score_file:
    :param corpus_dict:
    :param is_mz_score_file:
    :param example_file:
    :return:
    '''
    reference_file = '../../../data/twitter/ModelInput/test.tok.lc.response' # reference file (should do tokenization and lower case in advance)
    context_file = '../../../data/twitter/ModelInput/test.tok.lc.context' # context file
    response_generation_example = example_file
    test_id_to_metrics_dict = dict() # key: test instance id  / value: a dict storing BLUE/ROUGE-L metrics
    if is_mz_score_file == 1:
        test_id_to_gen_seq, test_id_selected_response_id = init_test_id_to_gen_seq(mz_score_file, corpus_dict)
    else:
        test_id_to_gen_seq = read_gen_resp(mz_score_file)
    print('test print test_id_to_gen_seq: ', test_id_to_gen_seq)
    test_id_to_context = read_hyps(context_file)
    test_ids_num = len(test_id_to_gen_seq)
    metrics_keys = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L',] #  'METEOR'  # Sentence BLEU, ROUGE-L, METEOR
    metrics = np.zeros((test_ids_num, len(metrics_keys))) # number of test instances * 6 metrics (BLUE-1 to BLUE-4 and ROUGE-L, METEOR)
    index =  -1
    gen_response_avg_len = 0.0
    gen_response_num = 0.0
    list_of_references = []
    hypotheses = []
    gen_response_words_corpus = [] # list of all words in the generated responses in the corups level
    with open(reference_file) as fin, open(response_generation_example, 'w') as fout:
        for l in tqdm(fin):
            index += 1
            gt_response = l.strip()
            gen_response = test_id_to_gen_seq['test_' + str(index)].strip()
            # here we do post-processing for _ _ mention _ _ and _ _ url _ _ to avoid bais on the computation of BLUE score
            gt_response = post_process(gt_response)
            gen_response = post_process(gen_response)
            gen_response_words = gen_response.lower().split(' ') #nltk.word_tokenize(gen_response.lower()) already tokenized
            gt_response_words = gt_response.lower().split(' ') #nltk.word_tokenize(gt_response.lower()) already tokenized
            gen_response_avg_len += len(gen_response_words)
            hypotheses.append(gen_response_words)
            list_of_references.append([gt_response_words])
            gen_response_words_corpus.extend(gen_response_words)
            gen_response_num += 1.0
            input_context = test_id_to_context[index].strip()
            # id \t context \t gt_response \t gen_response \t gen_response_id
            fout.write('test-' + str(index) + '\t****\t' +  post_process(input_context) + '\t****\t' + gt_response.strip() + '\t****\t' + gen_response  \
                       + '\t****\t' + test_id_selected_response_id['test_' + str(index)] + '\n')
            test_id_to_metrics_dict[index] = score_given_a_sequence_pair(gt_response.lower(), gen_response.lower()) # Sentence BLEU
            for m_i in range(0,len(metrics_keys)):
                metrics[index, m_i] = test_id_to_metrics_dict[index][metrics_keys[m_i]]

    avg_metrics = np.mean(metrics, axis=0)
    avg_metrics = [i * 100 for i in avg_metrics.tolist()]
    avg_len = gen_response_avg_len / gen_response_num
    BLEUscore_Corpus = nltk.translate.bleu_score.corpus_bleu(list_of_references, hypotheses) * 100
    dist1 = compute_distinct_n_gram(gen_response_words_corpus, 1)
    dist2 = compute_distinct_n_gram(gen_response_words_corpus, 2)
    res_string = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (avg_metrics[0], avg_metrics[1], avg_metrics[2] \
                 , avg_metrics[3], BLEUscore_Corpus, avg_metrics[4], dist1, dist2, avg_len)
    out_l =  res_string
    print 'out_l : ', out_l

def init_corpus_dict(corpus_file, corus_dict_text_start_col):
    corpus_dict = {}
    with open(corpus_file) as fin:
        for l in fin:
            tok = l.split(' ')
            # print('test tok: ', tok)
            corpus_dict[tok[0]] = ' '.join(tok[corus_dict_text_start_col:])
    return corpus_dict

if __name__=='__main__':
    data_name = 'twitter'
    if __name__ == '__main__':
        if len(sys.argv) < 6:
            print 'please input params: score_file corpus_file is_mz_score_file(1 or 0) ' \
                  'corus_dict_text_start_col (1 for fixed setting or 2 for original setting) example_file'
            exit(1)

    # mz_score_file = '../../../data/' + data_name + '/ModelRes/seq2seq-mix-ret/seq2seq-mix-ret-matchpyramid.predict-test.60'
    # corpus_file = '../../../data/' + data_name + '/ModelInput-seq2seq-mix-ret/corpus.txt'
    # two types of score file
    # mz_score: in TREC_EVAL format, we can get the generated response with corpus file
    # hypres file: just generated response. one query per line.
    score_file = sys.argv[1]
    corpus_file = sys.argv[2]
    is_mz_score_file = int(sys.argv[3])
    corus_dict_text_start_col = int(sys.argv[4])
    example_file = sys.argv[5]
    corpus_dict = init_corpus_dict(corpus_file, corus_dict_text_start_col)
    compute_bleu_rouge_given_single_file(score_file, corpus_dict, is_mz_score_file,
                                         example_file)



