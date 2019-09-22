'''
The script for data preprocess of the hybrid neural re-ranker of the hybrid neural
conversational model
The input data format is 'label \t context \t response \t qid \t did'
Add qid and did to avoid re-generate new qid and did to avoid problems in the future
'''


# /bin/python2.7
import os
import sys
sys.path.append('../../matchzoo/inputs/')
sys.path.append('../../matchzoo/utils/')

reload(sys)
sys.setdefaultencoding('utf8')

from preparation import Preparation
from preprocess import Preprocess, NgramUtil

def read_dict(infile):
    word_dict = {}
    for line in open(infile):
        r = line.strip().split()
        word_dict[r[1]] = r[0]
    return word_dict

def run_with_train_valid_test_corpus_given_qid_did_gen_unique_id_for_genres(
    train_file, valid_file, test_file):
    '''
    Run with pre-splited train_file, valid_file, test_file
    the qids and dids are also already given
    The input format should be label \t text1 \t text2 \t qid \t did
    The qid and did are given to avoid re-generate qid and did
    !!! But for the hybrid-ncm project, only the query ids and id for the retrieved
    results ids are unique. All the id for the generated results is "gen-res-0"
    Here to fix this bug, we further generate unique ids for all the generated
    results like "gen-res-0-newid-x"
    :param train_file: train file
    :param valid_file: valid file
    :param test_file: test file
    :return: corpus, rels_train, rels_valid, rels_test
    '''
    corpus = {}
    rels = []
    rels_train = []  # label qid did
    rels_valid = []
    rels_test = []
    gen_result_dict = {} # gen_result -> gen_result_id
    cur_gen_res_id = 0
    # merge corpus files, but return rels for train/valid/test seperately
    for file_path in list([train_file, valid_file, test_file]):
        if file_path == train_file:
            rels = rels_train
        elif file_path == valid_file:
            rels = rels_valid
        if file_path == test_file:
            rels = rels_test
        f = open(file_path, 'r')
        for line in f:
            line = line.decode('utf8')
            line = line.strip()
            label, t1, t2, qid, did = line.split('\t')
            if did == 'gen-res-0':
                # generate new did for this generated response
                if t2 in gen_result_dict:
                    did = gen_result_dict[t2]
                else:
                    did = 'gen-res-0-newid-' + str(cur_gen_res_id)
                    cur_gen_res_id += 1
                    gen_result_dict[t2] = did
            corpus[qid] = t1
            corpus[did] = t2
            rels.append((label, qid, did))
        f.close()
    print('generated new unique ids for these number of generated responses : ',
          len(gen_result_dict))
    return corpus, rels_train, rels_valid, rels_test

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'please input params: basedir need_preprocess (1 need/ 0 no need. default 0) save_space (1 need/ 0 no need. default 1)'
        exit(1)

    basedir = sys.argv[1]
    need_preprocess = sys.argv[2]
    save_space = sys.argv[3]
    # basedir = '../../../data/twitter/ModelInput-seq2seq-facts-mix-ret/'

    # transform context/response pairs into corpus file and relation file
    # the input files are train.txt/valid.txt/test.txt
    # the format of each line is 'label \t context \t response'
    prepare = Preparation()
    # run with three files (train.txt.mz, valid.txt.mz, test.txt.mz) to generate unique ids
    # for q/d in train/valid/test data. Since we will merge these three corpus files into a single file later
    corpus, rels_train, rels_valid, rels_test = run_with_train_valid_test_corpus_given_qid_did_gen_unique_id_for_genres(
        basedir + 'train.mz', basedir + 'valid.mz', basedir + 'test.mz')

    for data_part in list(['train', 'valid', 'test']):
        if data_part == 'train':
            rels = rels_train
        elif data_part == 'valid':
            rels = rels_valid
        else:
            rels = rels_test
        print 'total relations in ', data_part, len(rels)
        prepare.save_relation(basedir + 'relation_' + data_part + '.txt', rels)
        if save_space == '0':
            print 'filter queries with duplicated doc ids...'
            prepare.check_filter_query_with_dup_doc(basedir + 'relation_' + data_part + '.txt')
    print 'total corpus ', len(corpus)
    if save_space == '0':
        prepare.save_corpus(basedir + 'corpus.txt', corpus)
    print 'preparation finished ...'

    if need_preprocess == '1':
        print 'begin preprocess...'
        # Prerpocess corpus file
        preprocessor = Preprocess(word_filter_config={'min_freq': 2})
        dids, docs = preprocessor.run(basedir + 'corpus.txt')
        preprocessor.save_word_dict(basedir + 'word_dict.txt')
        # preprocessor.save_words_df(basedir + 'word_df.txt')

        fout = open(basedir + 'corpus_preprocessed.txt','w')
        for inum,did in enumerate(dids):
            fout.write('%s\t%s\t%s\n' % (did, len(docs[inum]), ' '.join(map(str, docs[inum])))) # id text_len text_ids
        fout.close()
        print('preprocess finished ...')

