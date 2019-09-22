## Introduction

This folder contains the implementation of the hybrid ranking module of the hybrid neural conversation model proposed in the CIKM'19 paper "A Hybrid Retrieval-Generation Neural Conversation Model". The implementation is based on [MatchZoo](https://github.com/NTMC-Community/MatchZoo).

If you use this code for your paper, please cite it as:

```
Liu Yang, Junjie Hu, Minghui Qiu, Chen Qu, Jianfeng Gao, W. Bruce Croft, Xiaodong Liu, Yelong Shen, Jingjing Liu. A Hybrid Retrieval-Generation Neural Conversation Model. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM 2019).

Bibtext
 @inproceedings{HybridNCM-CIKM19,
	author = {Yang, L. and Hu, J. and Qiu, M. and Qu, C. and Gao, J. and Croft, W. B. and Liu, X. and Shen, Y. and Liu, J.},
	title = {A Hybrid Retrieval-Generation Neural Conversation Model},
	booktitle = {CIKM '19},
	year = {2019},
}
```

## Requirements

* python 2.7
* tensorflow 1.2+
* keras 2.06+
* nltk 3.2.2+
* tqdm 4.19.4+
* h5py 2.7.1+

## Guide To Use

#### Data Preparation and Preprocess ####

We can not open source the Twitter/ Foursqure data used in our CIKM 2019 paper since it is an internal data set in Microsoft Research. To demo how to train and test the hybrid rankin module, we provide demo data generated based on the [Ubuntu Dialog Corpus](https://arxiv.org/abs/1506.08909) data in the folder HybridNCM/demo_data. You can easily adapt these instructions to other data sets in your lab or company.

* Step 1: Download and prepare the input data for the hybrid ranking module. you can download the Ubuntu Dialog Corpus(UDC) demo data from
[the Google Drive folder](https://drive.google.com/drive/u/0/folders/14kp-q1nre-mKjO4ExfXAVyhuGK3woxsq). This folder contains the processed context/response/facts for the train/dev/test data generated from the UDC data sets to train the Seq2Seq and Seq2Seq-Fact models. It also contains the generated responses and retrieved responses for the train/dev/test data, which can be input of the hybrid ranking module.

* Step 2: Mix the retrieved responses and generated responses for the same dialog context. Compute the distant supervision scores for response candidates like BLUE-1, BLEU-2 or ROUGE-L. The script hybrid_ranking/matchzoo/conqa/transfer_to_mz_format_hncm_hybrid_ranking.py will help you do these steps.

```
python transfer_to_mz_format_hncm_hybrid_ranking.py --help
usage: transfer_to_mz_format_hncm_hybrid_ranking.py [-h]
                                                    [--ret_res_file RET_RES_FILE]
                                                    [--gen_res_file GEN_RES_FILE]
                                                    [--context_file CONTEXT_FILE]
                                                    [--response_file RESPONSE_FILE]
                                                    [--mz_model_input_folder MZ_MODEL_INPUT_FOLDER]
                                                    [--data_partition DATA_PARTITION]

optional arguments:
  -h, --help            show this help message and exit
  --ret_res_file RET_RES_FILE
                        Ret_res_file: path of the retrieval result file
  --gen_res_file GEN_RES_FILE
                        Gen_res_file: path of the generation result file
  --context_file CONTEXT_FILE
                        Context_file: path of the context file
  --response_file RESPONSE_FILE
                        Response_file: path of the response file
  --mz_model_input_folder MZ_MODEL_INPUT_FOLDER
                        Mz_model_input_folder: path of the input folder of
                        MatchZoo
  --data_partition DATA_PARTITION
                        Data_partition: data partition train/dev/test

```

all of the input files like ret_res_file, gen_res_file, context_file, response_file has been uploaded to [the Google Drive folder](https://drive.google.com/drive/u/0/folders/14kp-q1nre-mKjO4ExfXAVyhuGK3woxsq). You can set mz_model_input_folder as the input data folder of [MatchZoo](https://github.com/NTMC-Community/MatchZoo) in your workspace.

* Step 3: Transfer the raw distant supervision score to the binary label for the training of re-ranker. Rank the response candidates according to the distant supervision score and select top K resopnse candidates as the positive response candidates and the other response candidates are the negative candidates. The script hybrid_ranking/matchzoo/conqa/transfer_to_mz_format_hncm_binary_label.py will help you do these steps.

```
python transfer_to_mz_format_hncm_binary_label.py --help
usage: transfer_to_mz_format_hncm_binary_label.py [-h]
                                                  [--pos_resp_candidate_num POS_RESP_CANDIDATE_NUM]
                                                  [--supervision_score SUPERVISION_SCORE]
                                                  [--mix_file MIX_FILE]
                                                  [--score_file SCORE_FILE]
                                                  [--mz_file MZ_FILE]
                                                  [--data_partition DATA_PARTITION]

optional arguments:
  -h, --help            show this help message and exit
  --pos_resp_candidate_num POS_RESP_CANDIDATE_NUM
                        Pos_resp_candidate_num: number of positive response
                        candidate per query
  --supervision_score SUPERVISION_SCORE
                        Supervision_score: type of distance supervision score.
                        The types inclue ROUGE_L, Bleu_1 and Bleu_2
  --mix_file MIX_FILE   Mix_file: path of the mix file which contains all the
                        retreived responses and generated responses for the
                        dialog contexts
  --score_file SCORE_FILE
                        Score_file: path of the score file which contains the
                        distanct supervsion scores for all response candidates
  --mz_file MZ_FILE     Mz_file: the generated file as the input data of
                        MatchZoo
  --data_partition DATA_PARTITION
                        Data_partition: data partition train/dev/test

```

* Step 4: Run data preprocessing steps in the MatchZoo toolkit. We process the input data train.mz/valid.mz/test.mz files and generate the relation files, corpus files, word dictionay files and preprocessed corpus files. Users can refer to the readme files of [MatchZoo](https://github.com/NTMC-Community/MatchZoo/tree/1.0) and [NeuralResponseRanking](https://github.com/yangliuy/NeuralResponseRanking) for all the details on how to do the data preprocessing for MatchZoo. To generate the relation files, corpus files, word dictionay files and preprocessed corpus files, you can refer to hybrid_ranking/matchzoo/conqa/preprocess_hncm.py. We used the pretrained Glove embedding. To generate the filtered version of Glove embedding file, you can refer to hybrid_ranking/matchzoo/conqa/gen_w2v_filtered.py. We uploaded some processed files into [the Google Drive folder](https://drive.google.com/drive/u/0/folders/14kp-q1nre-mKjO4ExfXAVyhuGK3woxsq) to help you understand the input files and output files in this step. Note that these files are generated based on the UDC data and only for demo purpose.

#### Training ####

The script for both training and testing of the hybrid ranking module is hybrid_ranking/matchzoo/main_hybrid_ncm.py. The supported parameters are as follows:

```
python main_hybrid_ncm.py --help
Using TensorFlow backend.
usage: main_hybrid_ncm.py [-h] [--phase PHASE] [--model_file MODEL_FILE]
                          [--embed_size EMBED_SIZE] [--embed_path EMBED_PATH]
                          [--weights_file WEIGHTS_FILE]
                          [--save_path SAVE_PATH]
                          [--train_relation_file TRAIN_RELATION_FILE]
                          [--valid_relation_file VALID_RELATION_FILE]
                          [--test_relation_file TEST_RELATION_FILE]
                          [--predict_relation_file PREDICT_RELATION_FILE]
                          [--test_weights_iters TEST_WEIGHTS_ITERS]
                          [--save_path_during_train SAVE_PATH_DURING_TRAIN]
                          [--text1_maxlen TEXT1_MAXLEN]
                          [--text2_maxlen TEXT2_MAXLEN]
                          [--learning_rate LEARNING_RATE] [--margin MARGIN]
                          [--dpool_size DPOOL_SIZE]
                          [--dropout_rate DROPOUT_RATE]
                          [--kernel_count KERNEL_COUNT]
                          [--kernel_size KERNEL_SIZE]
                          [--is_save_weights IS_SAVE_WEIGHTS]

optional arguments:
  -h, --help            show this help message and exit
  --phase PHASE         Phase: Can be train or predict, the default value is
                        train.
  --model_file MODEL_FILE
                        Model_file: MatchZoo model file for the chosen model.
  --embed_size EMBED_SIZE
                        Embed_size: number of dimensions in word embeddings.
  --embed_path EMBED_PATH
                        Embed_path: path of embedding file.
  --weights_file WEIGHTS_FILE
                        weights_file: path of weights file
  --save_path SAVE_PATH
                        save_path: path of predicted score file
  --train_relation_file TRAIN_RELATION_FILE
                        train relation file
  --valid_relation_file VALID_RELATION_FILE
                        valid relation file
  --test_relation_file TEST_RELATION_FILE
                        test relation file
  --predict_relation_file PREDICT_RELATION_FILE
                        predict relation file
  --test_weights_iters TEST_WEIGHTS_ITERS
                        iteration number of test weights
  --save_path_during_train SAVE_PATH_DURING_TRAIN
                        path of predicted score files during training
  --text1_maxlen TEXT1_MAXLEN
                        max length of query
  --text2_maxlen TEXT2_MAXLEN
                        max length of doc
  --learning_rate LEARNING_RATE
                        learning rate
  --margin MARGIN       margin in ranking hinge loss
  --dpool_size DPOOL_SIZE
                        size of pooling kernal
  --dropout_rate DROPOUT_RATE
                        dropout rate
  --kernel_count KERNEL_COUNT
                        count of conv kernals
  --kernel_size KERNEL_SIZE
                        size of conv kernal
  --is_save_weights IS_SAVE_WEIGHTS
                        whether want to save weights
```

It is developed based on the main file of MatchZoo. We added command flags for the hyper-parameters of MatchPyramid model we use in our paper. We also added the functions for computing the metrics like corpus level BLEU, ROUGE-L, Dist-1, Dist-2 reported in our paper.

To start the model training, you can run

```
python main_hybrid_ncm.py --phase train --model_file config/udc/matchpyramid-seq2seq-ret.config
```

Note that you need to update the path "../../../data/udc/google-drive-hybrid-ranking-input-demo/" in the model file to the corresponding path in your workspace.

The sample output for model training is as follows:


```
[09-01-2019 23:58:38]	[Train:train] Iter:0	loss=1.834167
[09-01-2019 23:59:15]	[Train:train] Iter:1	loss=1.732865
[09-01-2019 23:59:50]	[Train:train] Iter:2	loss=1.475936
[09-02-2019 00:00:22]	[Train:train] Iter:3	loss=1.281274
[09-02-2019 00:00:54]	[Train:train] Iter:4	loss=1.251016
[09-02-2019 00:01:28]	[Train:train] Iter:5	loss=1.141776
[09-02-2019 00:02:01]	[Train:train] Iter:6	loss=1.099355
[09-02-2019 00:02:34]	[Train:train] Iter:7	loss=1.046148
[09-02-2019 00:03:08]	[Train:train] Iter:8	loss=1.020757
[09-02-2019 00:03:40]	[Train:train] Iter:9	loss=1.001285
[09-02-2019 00:04:03]	[Eval:test] Iter:10	(bleu1-4 corpus_bleu rougel dist1 dist2 avglen)	7.1175	3.0661	2.5857	2.4381	4.0659	8.4367	0.0598	0.4528	16.8945
[09-02-2019 00:04:25]	[Eval:valid] Iter:10	(bleu1-4 corpus_bleu rougel dist1 dist2 avglen)	7.3899	3.2745	2.8079	2.7091	4.0026	8.7137	0.0611	0.4542	16.9840
......
```

#### Testing ####

After the model training finished, you will find the stored model weight files under the specified path in the model config file. The default path is "../../../data/udc/google-drive-hybrid-ranking-input-demo/model-res/". You can change this path to other paths in your workspace. To start model testing, you can run
```
python main_hybrid_ncm.py --phase predict --model_file config/udc/matchpyramid-seq2seq-ret.config
```

We will compute the metrics like BLEU-1 to BLEU-4, ROUGE-L etc during the model training process. You can also compute the metrics after model testing. The script hybrid_ranking/matchzoo/conqa/compute_metrics_given_predict_score_file.py will help you compute these metrics. The code for BLEU and ROUGE metrics are from the [evaluation codes for MS COCO caption generation.](https://github.com/tylin/coco-caption.git).