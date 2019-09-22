'''
Generate train/valid/test data for the hybrid ranking
module in HNCM based on matchzoo

The input data format of matchzoo is:
Label \t Context (maybe seperated by \t depends on the setting) \t Response_Candidates

The labels come from BLEU scores comparing with the ground truth response
Top K positive binary label transformation the transfer the data to the same
 format of the MatchZoo toolkit

Compute the BLEU/ROUGE-L score for the top 9 retrieval results and the 1 generated
result. Then use the top K candidates by BLEU/ROUGE-L score as the positive response candidates (label = 1)
and the other candidates as the negative response candidates (label = 0)

Generate *.txt files

Try to use ROUGE-L/BLEU-1/BLEU-2 based distant supervision signal for the experiments

'''
import os
import sys
from tqdm import tqdm
import nltk # to compute sentence_bleu
sys.path.append('../../metrics/')
from eval_test import score_given_a_sequence_pair # to compute rougel/bleu1/bleu2/bleu3/bleu4 (sentence level)
import argparse

# scorers = [
#         (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
#         #(Meteor(), "METEOR"), # hidde currently due to slow speed
#         (Rouge(), "ROUGE_L")
#         #(Cider(), "CIDEr")
#     ]

def compute_supervision_score(resp_gt, resp_gen, supervision_score_type):
    sentBLEU = compute_sentence_bleu_given_a_pair(resp_gt, resp_gen)
    score_dict = score_given_a_sequence_pair(resp_gt, resp_gen)
    score_dict['sentBLEU'] = sentBLEU
    score = score_dict[supervision_score_type]
    return str(score), score_dict

# here we do post-processing for _ _ mention _ _ and _ _ url _ _ to avoid bias on the computation of BLUE score
# transfer &apos; to '
# transfer to lower case
def post_process(ori_response):
    processed_response = ori_response.replace('_ _ mention _ _', 'mentionplaceholder',).replace('_ _ url _ _', 'urlplaceholder')
    processed_response = processed_response.replace('&apos;', "'").lower()
    return processed_response

def compute_sentence_bleu_given_a_pair(ref, hyp):
    hypothesis = post_process(ref).split(' ') # already tokenized
    reference = post_process(hyp).split(' ') # already tokenized
    # there may be several references
    return str(nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)*100)

def write_ret_response(fout, fout_super_score_dict, ret_l, query_context, query_response, score, supervision_score_type,
                       tok, score_dict):
    fout.write(ret_l.strip() + '\t' + query_context + '\t' + query_response + '\t' + score \
               + '\t' + supervision_score_type + '\n')  # add retrieved results
    fout_super_score_dict.write(tok[0] + '\t' + tok[1] + '\tROUGE_L\t' + str(score_dict['ROUGE_L']) \
                                + '\tBleu_1\t' + str(score_dict['Bleu_1']) + '\tBleu_2\t' + str(score_dict['Bleu_2'])
                                + '\tsentBLEU\t' + str(score_dict['sentBLEU']) + '\n')

def write_gen_response(fout, fout_super_score_dict, query_id, gen_response, query_context, query_response, score, supervision_score_type, score_dict):
    fout.write(query_id + '\tgen-res-0\tnull\t' + gen_response + '\t' + query_context \
               + '\t' + query_response + '\t' + score + '\t' + supervision_score_type + '\n')  # add generated results
    fout_super_score_dict.write(query_id + '\t' + 'gen-res-0' + '\tROUGE_L\t' + str(score_dict['ROUGE_L']) \
                                + '\tBleu_1\t' + str(score_dict['Bleu_1']) + '\tBleu_2\t' +  str(score_dict['Bleu_2'])
                                + '\tsentBLEU\t' + str(score_dict['sentBLEU']) + '\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ret_res_file',
                        default='../../../../data/udc/ModelInput/dev_retrieved_crpairs_seach_contextText.txt',
                        help='Ret_res_file: path of the retrieval result file')
    parser.add_argument('--gen_res_file',
                        default='../../../../data/udc/ModelRes/seq2seq_es_512_el_2_dl_2_hs_256_lr_0.001_lrd_0.5_vi_5000_pa_10/dev-decode-len100',
                        help='Gen_res_file: path of the generation result file')
    parser.add_argument('--context_file',
                        default='../../../../data/udc/ModelInput/dev.tok.lc.context',
                        help='Context_file: path of the context file')
    parser.add_argument('--response_file',
                        default='../../../../data/udc/ModelInput/dev.tok.lc.response',
                        help='Response_file: path of the response file')
    parser.add_argument('--mz_model_input_folder',
                        default='../../../../data/udc/ModelInput-seq2seq-mix-ret-demo/',
                        help='Mz_model_input_folder: path of the input folder '
                             'of MatchZoo')
    parser.add_argument('--data_partition',
                        default='dev',
                        help='Data_partition: data partition train/dev/test')

    args = parser.parse_args()
    ret_res_file = args.ret_res_file
    gen_res_file = args.gen_res_file
    context_file = args.context_file
    response_file = args.response_file
    model_input_folder = args.mz_model_input_folder
    data_partition = args.data_partition
    supervision_score_type = 'ROUGE_L' # ROUGE_L or Bleu_1 or Bleu_2 or sentBLEU (sent-BLEU is not recommended.)
    ret_response_num = 9 # number of retrieval candidates per queries

    # format of ret_res_file:  query_id \t doc_id \t retrieved_context \t retrieved_response
    # format of output_mix_file *.txt file:  query_id \t doc_id \t retrieved_context \t retrieved_response \t query_context \t query_response \t score \t score_label
    if not os.path.exists(model_input_folder):
        os.makedirs(model_input_folder)
        print('created folder :' + model_input_folder)

    gen_res_lines = open(gen_res_file).readlines()
    context_lines = open(context_file).readlines()
    response_lines = open(response_file).readlines()
    output_mix_file = model_input_folder + data_partition + '.txt'
    output_supervision_score_file = model_input_folder + data_partition \
                                    + '.supervision_score_dict'
    print('generated file for the mix response file: ', output_mix_file)
    print('generated file for the supervision score file : ',
          output_supervision_score_file)
    with open(ret_res_file) as f_ret_in , open(output_mix_file, 'w') as fout, \
         open(output_supervision_score_file, 'w') as fout_super_score_dict:
        cur_qid = 'init'
        cur_cand_num = 0
        for ret_l in tqdm(f_ret_in):
            tok = ret_l.split('\t')
            instance_id = int(tok[0].split('_')[1])
            query_context = context_lines[instance_id].strip()
            query_response = response_lines[instance_id].strip()
            if cur_qid == tok[0]: # same query
                if cur_cand_num < ret_response_num:
                    score, score_dict = compute_supervision_score(query_response, tok[3].strip(), supervision_score_type)
                    write_ret_response(fout, fout_super_score_dict, ret_l, query_context, query_response, score,
                                       supervision_score_type, tok, score_dict)
                    cur_cand_num += 1
                elif cur_cand_num == ret_response_num:
                    score, score_dict = compute_supervision_score(query_response, gen_res_lines[instance_id].strip(), supervision_score_type)
                    write_gen_response(fout, fout_super_score_dict, tok[0], gen_res_lines[instance_id].strip(), query_context,
                                       query_response, score, supervision_score_type, score_dict)
                    cur_cand_num += 1
                else:
                    continue
            else: # new query
                if cur_cand_num < ret_response_num + 1 and cur_qid != 'init':
                    # print('found specific query context with no hits in retreival results : cur_qid = ', cur_qid)
                    # print('cur_cand_num = ', cur_cand_num)
                    cur_qid_int = int(cur_qid.split('_')[1])
                    score, score_dict = compute_supervision_score(response_lines[cur_qid_int].strip(),gen_res_lines[cur_qid_int].strip(), supervision_score_type)
                    write_gen_response(fout, fout_super_score_dict, cur_qid, gen_res_lines[cur_qid_int].strip(),
                                       context_lines[cur_qid_int].strip(), response_lines[cur_qid_int].strip(),
                                       score, supervision_score_type, score_dict)
                cur_qid = tok[0]
                score, score_dict = compute_supervision_score(query_response, tok[3].strip(), supervision_score_type)
                #print('test score : ', score)
                #print('test supervision_score_type: ', supervision_score_type)
                write_ret_response(fout, fout_super_score_dict, ret_l, query_context, query_response, score,
                                   supervision_score_type, tok, score_dict)
                cur_cand_num = 1



