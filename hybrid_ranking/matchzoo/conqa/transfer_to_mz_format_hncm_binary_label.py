'''
Generate train/valid/test data for the hybrid ranking
module in HNCM based on matchzoo

The input data format of matchzoo is:
Label \t Context (maybe seperated by \t depends on the setting) \t Response_Candidates \t Qid \t Did

Add Qid and Did to avoid re-generate Qid and Did

The labels come from BLEU scores comparing with the ground truth response
Top K positive binary label transformation the transfer the data to the same format of
the MatchZoo toolkit

Compute the BLEU score for the top 9 retrieval results and the 1 generated
result. Then use the top K candidates by BLEU score as the positive response candidates (label = 1)
and the other candidates as the negative response candidates (label = 0)

This script is for transfering the raw BLEU to the binary label for the training of re-ranker

Generate *.mz files

'''

import numpy as np
from transfer_to_mz_format_hncm_hybrid_ranking import post_process
import argparse

def generate_mz_file(mix_file, score_file, mz_file, max_context_query_num,
                     pos_resp_candidate_num, supervision_score):
    # format of mix_file query_id \t doc_id \t retrieved_context \t retrieved_response \t query_context \t query_response \it supervision_score
    # format of mz_file label \t context \t candidate responses \t query_id \t doc_id
    supervision_score_index_dict = {
        'ROUGE_L': 3,
        'Bleu_1': 5,
        'Bleu_2': 7,
        'sentBLEU': 9
    }
    context_query_num = 0
    with open(mix_file) as fin_mix, open(mz_file, 'w') as fout, open(score_file) as fin_score:
        cur_qid = 'init'
        cached_pairs = []
        cached_scores = []
        print('generate file ', mz_file)
        for mix_l in fin_mix:
            toks = mix_l.split('\t')
            score_toks = fin_score.readline().strip().split('\t')
            if score_toks[0] != toks[0] or score_toks[1] != toks[1]:
                raise Exception('Find a score line with non-consistant qid/did (score_toks): ', score_toks)
            if toks[1] == 'NULL' and toks[2] == 'NULL' and toks[3] == 'NULL' and toks[4] == 'NULL':
                continue # no retrieved response candidates for this query context (the number of retrieved results could be 0)
            # print('toks: ', toks)
            query_context = post_process(toks[4])
            candidate_response = post_process(toks[3])
            raw_score = float(score_toks[supervision_score_index_dict[supervision_score]])
            if cur_qid == toks[0]: # same query
                if remove_train_self_candidate and data_partition == 'train' and toks[0].split('_')[1] ==  toks[1]:
                    # this is a retrieved candidate which is the same with the query context/response
                    # print 'test toks: ', toks
                    continue
                cached_pairs.append(query_context + '\t' + candidate_response + '\t' + toks[0] + '\t' + toks[1])
                cached_scores.append(raw_score)
            else: # new query
                # sort cached_scores and transfer the top k scores to 1; the other scores to 0
                # print('test before cached_scores = ', cached_scores)
                top_k_index = np.array(cached_scores).argsort()[-pos_resp_candidate_num:][::-1]
                # print('test top_k_index = ', top_k_index)
                scores = np.zeros(len(cached_scores))
                scores[top_k_index] = 1
                # print('test after scores = ', scores)
                # output all lines for this query
                # if there is only one candidate for this query in the training data, skip it! (only for training data)
                if data_partition != 'train' or len(cached_scores) > 1:
                    for i in range(0, len(scores)):
                        fout.write(str(int(scores[i])) + '\t' + cached_pairs[i] + '\n')
                if remove_train_self_candidate and data_partition == 'train' and toks[0].split('_')[1] ==  toks[1]:
                    # this is a retrieved candidate which is the same with the query context/response
                    # print 'test toks: ', toks
                    cached_pairs = []
                    cached_scores = []
                else:
                    cached_pairs = [query_context + '\t' + candidate_response + '\t' + toks[0] + '\t' + toks[1]]
                    cached_scores = [raw_score]
                cur_qid = toks[0]
                context_query_num += 1
                if context_query_num > max_context_query_num: # * train_data_amount_ratio: # max 10 * max_context_query_num context/response pairs
                    break
        # the last query in the cache
        if data_partition != 'train' or len(cached_scores) > 1:
            top_k_index = np.array(cached_scores).argsort()[-pos_resp_candidate_num:][::-1]
            scores = np.zeros(len(cached_scores))
            scores[top_k_index] = 1
            # output all lines for this query
            for i in range(0, len(scores)):
                fout.write(str(int(scores[i])) + '\t' + cached_pairs[i] + '\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_resp_candidate_num',
                        default='2',
                        help='Pos_resp_candidate_num: number of positive '
                             'response candidate per query')
    parser.add_argument('--supervision_score',
                        default='ROUGE_L',
                        help='Supervision_score: type of distance supervision '
                             'score. The types inclue ROUGE_L, Bleu_1 and '
                             'Bleu_2')
    parser.add_argument('--mix_file',
                        default='../../../../data/udc/ModelInput-seq2seq-mix-ret-demo/dev.txt',
                        help='Mix_file: path of the mix file which contains '
                             'all the retreived responses and generated '
                             'responses for the dialog contexts')
    parser.add_argument('--score_file',
                        default='../../../../data/udc/ModelInput-seq2seq-mix-ret-demo/dev.supervision_score_dict',
                        help='Score_file: path of the score file which contains '
                             'the distanct supervsion scores for all response '
                             'candidates')
    parser.add_argument('--mz_file',
                        default='../../../../data/udc/ModelInput-seq2seq-mix-ret-demo/valid.mz',
                        help='Mz_file: the generated file as the input data of '
                             'MatchZoo')
    parser.add_argument('--data_partition',
                        default='dev',
                        help='Data_partition: data partition train/dev/test')
    args = parser.parse_args()
    pos_resp_candidate_num = int(args.pos_resp_candidate_num)
    supervision_score = args.supervision_score
    mix_file = args.mix_file
    score_file = args.score_file
    mz_file = args.mz_file
    data_partition = args.data_partition

    max_context_query_num = 1000000 # the number of training context/response pairs is 10 times of this number
    remove_train_self_candidate = False # whether remove the candidates which are the same with the query context/response pair
    generate_mz_file(mix_file, score_file, mz_file, max_context_query_num,
                     pos_resp_candidate_num, supervision_score)


