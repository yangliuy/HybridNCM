# -*- coding: utf8 -*-
import os
import sys
import time
import json
import argparse
import random

random.seed(49999)
import numpy
numpy.random.seed(49999)
import tensorflow
tensorflow.set_random_seed(49999)

from collections import OrderedDict

import keras
import keras.backend as K
from keras.models import Sequential, Model

from utils import *
import inputs
import metrics
from losses import *

sys.path.append('../metrics/')
from conqa.compute_metrics_given_predict_score_file import compute_bleu_rouge_given_scores_in_train, read_refs


def load_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    if model_type == 'JSON':
        mo = Model.from_config(config['model'])
    elif model_type == 'PY':
        model_config = config['model']['setting']
        model_config.update(config['inputs']['share'])
        sys.path.insert(0, config['model']['model_path'])

        model = import_object(config['model']['model_py'], model_config)
        mo = model.build()
    return mo


def train(config):

    print(json.dumps(config, indent=2))
    # read basic config
    global_conf = config["global"]
    optimizer = global_conf['optimizer']
    weights_file = str(global_conf['weights_file']) + '.%d'
    display_interval = int(global_conf['display_interval'])
    num_iters = int(global_conf['num_iters'])
    save_weights_iters = int(global_conf['save_weights_iters'])
    is_save_weights = global_conf['is_save_weights']

    # read input config
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    # prepare the corpus files and reference files for computing BLEU/ROUGE-L metrics
    corpus_file = share_input_conf['corpus_file']
    test_ref_list = read_refs(share_input_conf['test_ref_file'])
    valid_ref_list = read_refs(share_input_conf['valid_ref_file'])
    corpus_dict = {}
    with open(corpus_file) as fin:
        for l in fin:
            tok = l.split(' ')
            corpus_dict[tok[0]] = ' '.join(tok[2:])

    # collect embedding
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'])
        _PAD_ = share_input_conf['vocab_size'] - 1
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print '[Embedding] Embedding Load Done.'

    # list all input tags and construct tags config
    input_train_conf = OrderedDict()
    input_eval_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'TRAIN':
            input_train_conf[tag] = {}
            input_train_conf[tag].update(share_input_conf)
            input_train_conf[tag].update(input_conf[tag])
        elif input_conf[tag]['phase'] == 'EVAL':
            input_eval_conf[tag] = {}
            input_eval_conf[tag].update(share_input_conf)
            input_eval_conf[tag].update(input_conf[tag])
    print '[Input] Process Input Tags. %s in TRAIN, %s in EVAL.' % (input_train_conf.keys(), input_eval_conf.keys())

    # collect dataset identification
    dataset = {}
    for tag in input_conf:
        if tag != 'share' and input_conf[tag]['phase'] == 'PREDICT':
            continue
        if 'text1_corpus' in input_conf[tag]:
            datapath = input_conf[tag]['text1_corpus']
            if datapath not in dataset:
                dataset[datapath], _ = read_data(datapath)
        if 'text2_corpus' in input_conf[tag]:
            datapath = input_conf[tag]['text2_corpus']
            if datapath not in dataset:
                dataset[datapath], _ = read_data(datapath)
    print '[Dataset] %s Dataset Load Done.' % len(dataset)

    # initial data generator
    train_gen = OrderedDict()
    eval_gen = OrderedDict()

    for tag, conf in input_train_conf.items():
        print conf
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        train_gen[tag] = generator( config = conf )

    for tag, conf in input_eval_conf.items():
        print conf
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        eval_gen[tag] = generator( config = conf )

    output_conf = config['outputs']

    ######### Load Model #########
    model = load_model(config)

    loss = []
    for lobj in config['losses']:
        if lobj['object_name'] in mz_specialized_losses:
            loss.append(rank_losses.get(lobj['object_name'])(lobj['object_params']))
        else:
            loss.append(rank_losses.get(lobj['object_name']))
    eval_metrics = OrderedDict()
    for mobj in config['metrics']:
        mobj = mobj.lower()
        if '@' in mobj:
            mt_key, mt_val = mobj.split('@', 1)
            eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
        else:
            eval_metrics[mobj] = metrics.get(mobj)
    model.compile(optimizer=optimizer, loss=loss)
    print '[Model] Model Compile Done.'

    for i_e in range(num_iters):
        for tag, generator in train_gen.items():
            genfun = generator.get_batch_generator()
            print '[%s]\t[Train:%s]' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag),
            history = model.fit_generator(
                    genfun,
                    steps_per_epoch = display_interval,
                    epochs = 1,
                    shuffle=False,
                    verbose = 0
                ) #callbacks=[eval_map])
            print 'Iter:%d\tloss=%.6f' % (i_e, history.history['loss'][0])

        for tag, generator in eval_gen.items():
            #print('test tag: ', tag)
            genfun = generator.get_batch_generator()
            # print '[%s]\t[Eval:%s]' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag),
            # res = dict([[k,0.] for k in eval_metrics.keys()])
            res_scores = {} # 2D dict; key qid-did ;value: predict_score, ground_truth
            for input_data, y_true in genfun:
                y_pred = model.predict(input_data, batch_size=len(y_true))
                if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
                    list_counts = input_data['list_counts'] # list_counts store the boundries between documents under different queries
                    y_pred = np.squeeze(y_pred)
                    for lc_idx in range(len(list_counts) - 1):
                        pre = list_counts[lc_idx]
                        suf = list_counts[lc_idx + 1]
                        for p, y, t in zip(input_data['ID'][pre:suf], y_pred[pre:suf], y_true[pre:suf]):
                            if p[0] not in res_scores:
                                res_scores[p[0]] = {}
                            res_scores[p[0]][p[1]] = (y, t)
                else:
                    NameError('not supported in this version!')
            generator.reset()
            sys.stdout.flush()
            # save predicted score files for valid/test data
            if (i_e + 1) % save_weights_iters == 0:
                score_list = []
                with open(output_conf['predict']['save_path_during_train'] + '-' + tag + '.' + str(i_e + 1), 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d: d[1][0], reverse=True)
                        for inum, (did, (score, gt)) in enumerate(dinfo):
                            score_l = '%s\tQ0\t%s\t%d\t%f\t%s\t%s' % (qid, did, inum, score, config['net_name'], gt)
                            print >> f, score_l
                            score_list.append(score_l)
                # compute BLEU/ROUGE metrics at this check point
                ref_list = test_ref_list if tag == 'test' else valid_ref_list
                bleu_rouge_metrics = compute_bleu_rouge_given_scores_in_train(score_list,corpus_dict,ref_list,tag)
                print '[%s]\t[Eval:%s] Iter:%d\t(bleu1-4 corpus_bleu rougel dist1 dist2 avglen)\t%s' \
                    % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag, i_e+1, bleu_rouge_metrics)

        if (i_e+1) % save_weights_iters and is_save_weights == "1": # add an option to control saving weight files or not
            model.save_weights(weights_file % (i_e+1))

def predict(config):
    ######## Read input config ########

    print(json.dumps(config, indent=2))
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    # collect embedding
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'])
        _PAD_ = share_input_conf['vocab_size'] - 1
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.02, 0.02, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print '[Embedding] Embedding Load Done.'

    # list all input tags and construct tags config
    input_predict_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'PREDICT':
            input_predict_conf[tag] = {}
            input_predict_conf[tag].update(share_input_conf)
            input_predict_conf[tag].update(input_conf[tag])
    print '[Input] Process Input Tags. %s in PREDICT.' % (input_predict_conf.keys())

    # collect dataset identification
    dataset = {}
    for tag in input_conf:
        if tag == 'share' or input_conf[tag]['phase'] == 'PREDICT':
            if 'text1_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text1_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath)
            if 'text2_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text2_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath)
    print '[Dataset] %s Dataset Load Done.' % len(dataset)

    # initial data generator
    predict_gen = OrderedDict()

    for tag, conf in input_predict_conf.items():
        print conf
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        predict_gen[tag] = generator(
                                    #data1 = dataset[conf['text1_corpus']],
                                    #data2 = dataset[conf['text2_corpus']],
                                     config = conf )

    ######## Read output config ########
    output_conf = config['outputs']

    ######## Load Model ########
    global_conf = config["global"]
    weights_file = str(global_conf['weights_file']) + '.' + str(global_conf['test_weights_iters'])

    model = load_model(config)
    model.load_weights(weights_file)

    eval_metrics = OrderedDict()
    for mobj in config['metrics']:
        mobj = mobj.lower()
        if '@' in mobj:
            mt_key, mt_val = mobj.split('@', 1)
            eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
        else:
            eval_metrics[mobj] = metrics.get(mobj)
    res = dict([[k,0.] for k in eval_metrics.keys()])

    for tag, generator in predict_gen.items():
        genfun = generator.get_batch_generator()
        print '[%s]\t[Predict] @ %s ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag),
        num_valid = 0
        res_scores = {}
        for input_data, y_true in genfun:
            y_pred = model.predict(input_data, batch_size=len(y_true) )

            if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
                list_counts = input_data['list_counts']
                for k, eval_func in eval_metrics.items():
                    for lc_idx in range(len(list_counts)-1):
                        pre = list_counts[lc_idx]
                        suf = list_counts[lc_idx+1]
                        res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])

                y_pred = np.squeeze(y_pred)
                for lc_idx in range(len(list_counts)-1):
                    pre = list_counts[lc_idx]
                    suf = list_counts[lc_idx+1]
                    for p, y, t in zip(input_data['ID'][pre:suf], y_pred[pre:suf], y_true[pre:suf]):
                        if p[0] not in res_scores:
                            res_scores[p[0]] = {}
                        res_scores[p[0]][p[1]] = (y, t)

                num_valid += len(list_counts) - 1
            else:
                for k, eval_func in eval_metrics.items():
                    res[k] += eval_func(y_true = y_true, y_pred = y_pred)
                for p, y, t in zip(input_data['ID'], y_pred, y_true):
                    if p[0] not in res_scores:
                        res_scores[p[0]] = {} # 2D dict. The first key is qid; the second key is did
                    res_scores[p[0]][p[1]] = (y[1], t[1]) # value: pred_score, ground_truth_score
                num_valid += 1
        generator.reset()

        if tag in output_conf:
            if output_conf[tag]['save_format'] == 'TREC':
                with open(output_conf[tag]['save_path'], 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d:d[1][0], reverse=True)
                        for inum,(did, (score, gt)) in enumerate(dinfo):
                            print >> f, '%s\tQ0\t%s\t%d\t%f\t%s\t%s'%(qid, did, inum, score, config['net_name'], gt)
            elif output_conf[tag]['save_format'] == 'TEXTNET':
                with open(output_conf[tag]['save_path'], 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d:d[1][0], reverse=True)
                        for inum,(did, (score, gt)) in enumerate(dinfo):
                            print >> f, '%s %s %s %s'%(gt, qid, did, score)

        print '[Predict] results: ', '\t'.join(['%s=%f'%(k,v/num_valid) for k, v in res.items()])
        sys.stdout.flush()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', help='Phase: Can be train or predict, the default value is train.')
    parser.add_argument('--model_file', default='config/udc/matchpyramid-seq2seq-ret.config', help='Model_file: MatchZoo model file for the chosen model.')

    # optional parameters
    parser.add_argument('--embed_size', help='Embed_size: number of dimensions in word embeddings.')
    parser.add_argument('--embed_path', help='Embed_path: path of embedding file.')
    parser.add_argument('--weights_file', help='weights_file: path of weights file')
    parser.add_argument('--save_path', help='save_path: path of predicted score file')
    parser.add_argument('--train_relation_file', help='train relation file')
    parser.add_argument('--valid_relation_file', help='valid relation file')
    parser.add_argument('--test_relation_file', help='test relation file')
    parser.add_argument('--predict_relation_file', help='predict relation file')
    parser.add_argument('--test_weights_iters', help='iteration number of test weights')
    parser.add_argument('--save_path_during_train', help='path of predicted score files during training')

    # for hyper-parameters tuning using the matchpyramid model
    parser.add_argument('--text1_maxlen', help='max length of query')
    parser.add_argument('--text2_maxlen', help='max length of doc')
    parser.add_argument('--learning_rate', help='learning rate')
    parser.add_argument('--margin', help='margin in ranking hinge loss')
    parser.add_argument('--dpool_size', help='size of pooling kernal')
    parser.add_argument('--dropout_rate', help='dropout rate')
    parser.add_argument('--kernel_count', help=' count of conv kernals')
    parser.add_argument('--kernel_size', help='size of conv kernal')

    # add a parameter to control whether want to save weights
    parser.add_argument('--is_save_weights', default='1',
                        help='whether want to save weights')

    args = parser.parse_args()
    model_file =  args.model_file

    embed_size = args.embed_size
    embed_path = args.embed_path
    weights_file = args.weights_file
    save_path = args.save_path
    train_relation_file = args.train_relation_file
    valid_relation_file = args.valid_relation_file
    test_relation_file = args.test_relation_file
    predict_relation_file = args.predict_relation_file
    test_weights_iters = args.test_weights_iters
    save_path_during_train = args.save_path_during_train

    text1_maxlen = args.text1_maxlen
    text2_maxlen = args.text2_maxlen
    learning_rate = args.learning_rate
    margin = args.margin
    dpool_size = args.dpool_size
    dropout_rate = args.dropout_rate
    kernel_count = args.kernel_count
    kernel_size = args.kernel_size

    is_save_weights = args.is_save_weights

    with open(model_file, 'r') as f:
        config = json.load(f)
    if embed_size != None:
        config['inputs']['share']['embed_size'] = int(embed_size)
    if embed_path != None:
        config['inputs']['share']['embed_path'] = embed_path
    if weights_file != None:
        config['global']['weights_file'] = weights_file
    if test_weights_iters != None:
        config['global']['test_weights_iters'] = int(test_weights_iters)
    if save_path != None:
        config['outputs']['predict']['save_path'] = save_path
    if train_relation_file != None:
        config['inputs']['train']['relation_file'] = train_relation_file
    if valid_relation_file != None:
        config['inputs']['valid']['relation_file'] = valid_relation_file
    if test_relation_file != None:
        config['inputs']['test']['relation_file'] = test_relation_file
    if predict_relation_file != None:
        config['inputs']['predict']['relation_file'] = predict_relation_file
    if save_path_during_train != None:
        config['outputs']['predict']['save_path_during_train'] = save_path_during_train

    if text1_maxlen != None:
        config['inputs']['share']['text1_maxlen'] = int(text1_maxlen)
    if text2_maxlen != None:
        config['inputs']['share']['text2_maxlen'] = int(text2_maxlen)
    if learning_rate != None:
        config['global']['learning_rate'] = float(learning_rate)
    if margin != None:
        config['losses'][0]['object_params']['margin'] = float(margin)
    if dpool_size != None:
        config['model']['setting']['dpool_size'] = [int(dpool_size), int(dpool_size)]
    if dropout_rate != None:
        config['model']['setting']['dropout_rate'] = float(dropout_rate)
    if kernel_count != None:
        config['model']['setting']['kernel_count'] = int(kernel_count)
    if kernel_size != None:
        config['model']['setting']['kernel_size'] = [int(kernel_size),
                                                    int(kernel_size)]

    if is_save_weights != None:
        config['global']['is_save_weights'] = is_save_weights

    if args.phase == 'train':
        train(config)
    elif args.phase == 'predict':
        predict(config)
    else:
        print 'Phase Error.'
    return

if __name__=='__main__':
    main(sys.argv)
