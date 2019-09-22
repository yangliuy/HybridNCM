'''
Transfer Ubuntu Dialog Corpus to the same input data format with Twitter data
as the input data of HybridNCM model

1 Only use positive context/response pairs
2 To avoid too long context, transfer context to multiple (context, response)
pairs (u1,u2,u3,r) -> 3 context/response pairs (u1,u2), (u2,u3), (u3,r)
3 After do all these transformation steps for train/valid/test data, merge all
the (context, response) pairs into a single file (whole file for generating
vocab file), then re-generate the train/valid/test paritions by 8:1:1 or other
ratios
4 Compute text length statistics for the updated data set. The data statistics
property should be similar to Twitter/FourSquare data.

'''

from tqdm import tqdm

def merge_pos_pairs_into_dialog_file(data_folder, dialog_file):
    ''' Merge all positive pairs of the udc data into a single file
    :param data_folder: the input data folder which contains
    train.txt/valid.txt/test.txt
    :param dialog_file: the dialog file containing the merged dialogs
    :return: none
    '''
    fout = open(dialog_file, 'w', encoding="utf-8")
    for partition in list(['train', 'valid']): # only use the train/valid data
        # of the raw udc data sets
        with open(data_folder + partition + '.txt', encoding="utf-8") as fin:
            for l in tqdm(fin):
                tokens = l.split('\t')
                #print ('test num of cols in this line: ', len(tokens))
                if tokens[0] == '0':
                    continue # skip negative pairs
                fout.write(l)

def generate_train_valid_test_partitions(context_response_file, data_folder,
                                         max_train_pair_num,
                                         max_valid_pair_num,
                                         max_test_pair_num):
    '''
    :param context_response_file: context response file
    :param data_folder: data folder for udc
    :param max_train_pair_num: max number of train pairs
    :param max_valid_pair_num: max number of valid pairs
    :param max_test_pair_num: max number of test pairs
    :return: None
    '''
    train_partition_file = data_folder + 'train_context_response.txt'
    dev_partition_file = data_folder + 'dev_context_response.txt'
    test_partition_file = data_folder + 'test_context_response.txt'

    print('start generate train/valid/test partitions ...')
    with open(context_response_file, encoding='utf-8') as fin, open(
        train_partition_file, 'w', encoding='utf-8') as fout_train, \
        open(dev_partition_file, 'w', encoding='utf-8') as fout_dev, open(
        test_partition_file, 'w', encoding='utf-8') as fout_test:
        cur_line_index = 0
        for l in tqdm(fin):
            if cur_line_index < max_train_pair_num:
                fout_train.write(l)
            elif cur_line_index < max_train_pair_num + max_valid_pair_num:
                fout_dev.write(l)
            elif cur_line_index < max_train_pair_num + max_valid_pair_num + \
                  max_test_pair_num:
                fout_test.write(l)
            cur_line_index += 1
    print('generate train/valid/test partitions done!')

def generate_context_response_pairs(dialog_file, context_response_file, step,
                                    max_seq_len, min_seq_len):
    print('start generate context/response pairs...')
    # The input data format of UDC data is
    # label \t  utterance_1 \t utterance_2 \t  ...  \t candidate_response
    with open(dialog_file, encoding='utf-8' ) as fin, open(
        context_response_file, 'w', encoding='utf-8' ) as fout:
        for l in tqdm(fin):
            tokens = l.strip().split('\t')
            tokens = tokens[1:len(tokens)]
            # print('test tokens: ', tokens[0].encode('utf-8').strip())
            for i in range(0, len(tokens) - 1, step):
                # We ignore the last line (no answer for it)
                context = tokens[i]
                response = tokens[i + 1]
                pair = context + '\t' + response
                # 1. skip pairs with empty context or response
                if context.strip() == '' or response.strip() == '' or len(
                    pair.split('\t')) < 2:
                    continue
                # 2. skip context/response pairs which are longer than max
                # length or shorter than min length
                lenc = len(context.strip().split(' '))
                lenr = len(response.strip().split(' '))
                if lenc > max_seq_len or lenr > max_seq_len or \
                    lenc < min_seq_len or lenr < min_seq_len:
                    continue
                fout.write(pair + '\n')

if __name__ == '__main__':
    data_folder = '../../../data/udc/ModelInput/'
    dialog_file = data_folder + 'udc_dialog_whole.txt'
    context_response_file = data_folder + 'udc_context_response_whole.txt'

    # 1. Merge all the positive multi-turn/response pairs in train/valid/test
    # files into a single dialog file
    # The input data format of UDC data is
    # label \t  utterance_1 \t utterance_2 \t  ...  \t candidate_response
    # Only use the train/valid data of the original udc data
    merge_pos_pairs_into_dialog_file(data_folder, dialog_file)

    # 2. Generate context/response pairs from dialogs
    # step = 1  # suppose we have a conversation (a,b,c,d), we will generate
    # (a,b) (b,c) (c,d) as context/response pairs if step = 1; we will
    # generate (a,b), (c,d) as context/response pairs if step = 2 .
    # by default, we set step = 1 to generate more training data to train
    # seq2seq models
    step = 1
    max_seq_len = 100
    min_seq_len = 2
    generate_context_response_pairs(dialog_file, context_response_file, step,
                                    max_seq_len, min_seq_len)

    # 3. Generate train/valid/test partitions based on the whole context_r
    # esponse file
    max_train_pair_num = 1100000  # max number of pairs in train
    max_valid_pair_num = 10000  # max number of pairs in valid
    max_test_pair_num = 10000  # max number of pairs in test
    generate_train_valid_test_partitions(context_response_file, data_folder,
                                         max_train_pair_num,
                                         max_valid_pair_num,
                                         max_test_pair_num)