'''
Generate the train.tok.clean.lc.crpairs format: context ||| response
for UDC data to build the search index of context/response to run
the BM25 QQ match based "Only Retrieval" method

//grep '|||' train.tok.clean.lc.context
//grep '|||' train.tok.clean.lc.response
//There is no '|||' in the processed context/response file.
//So the format context ||| response is OK

'''

from tqdm import tqdm

if __name__ == '__main__':
    model_input_data_path = '../../../data/udc/ModelInput/'
    train_context_file = model_input_data_path + 'train.tok.clean.lc.context'
    train_response_file = model_input_data_path + 'train.tok.clean.lc.response'
    context_response_index_file = model_input_data_path + 'train.tok.clean.lc.crpairs'

    with open(train_context_file) as fin_c, open(train_response_file) as fin_r, \
        open(context_response_index_file, 'w') as fout:
        for l_c in tqdm(fin_c):
            context = l_c.strip()
            response = fin_r.readline()
            fout.write(context + "|||" + response)