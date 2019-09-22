# Introduction

This folder contains the implementation of the generation module of the hybrid neural conversation model proposed in the CIKM'19 paper "A Hybrid Retrieval-Generation Neural Conversation Model".

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

# Generation Module
=================
Implemented by [Junjie Hu](http://www.cs.cmu.edu/~junjieh/)

Contact: junjieh@cs.cmu.edu

## Requirements
- python 3.6
- pytorch 0.4.1
- numpy 1.16.3

## Get Started
The training data are stored under $repo/generation/data. To train the generation model, run the following commands. The models are saved under $repo/generation/model.

    cd $repo/generation
    bash train.sh [GPU id]

The sample output is as follows.

```
time 2019-09-22 18:41:41.036222, epoch 0, iter 50, avg. loss 6.696271, avg. ppl 809.38 example 1140, time elapsed 13.29 seconds
time 2019-09-22 18:41:54.030382, epoch 0, iter 100, avg. loss 6.140733, avg. ppl 464.39 example 1059, time elapsed 26.29 seconds
time 2019-09-22 18:42:06.552568, epoch 0, iter 150, avg. loss 6.128897, avg. ppl 458.93 example 898, time elapsed 38.81 seconds
time 2019-09-22 18:42:17.753383, epoch 0, iter 200, avg. loss 6.112555, avg. ppl 451.49 example 1003, time elapsed 50.01 seconds
time 2019-09-22 18:42:29.373735, epoch 0, iter 250, avg. loss 6.137636, avg. ppl 462.96 example 1156, time elapsed 61.63 seconds
time 2019-09-22 18:42:42.530632, epoch 0, iter 300, avg. loss 6.014463, avg. ppl 409.31 example 1131, time elapsed 74.79 seconds
time 2019-09-22 18:42:54.721629, epoch 0, iter 350, avg. loss 6.034187, avg. ppl 417.46 example 1045, time elapsed 86.98 seconds
time 2019-09-22 18:43:07.934838, epoch 0, iter 400, avg. loss 6.038451, avg. ppl 419.24 example 1092, time elapsed 100.19 seconds
time 2019-09-22 18:43:20.757709, epoch 0, iter 450, avg. loss 5.938180, avg. ppl 379.24 example 1091, time elapsed 113.02 seconds
......
```

Run the following command to generate the decoded responses.

    bash test.sh [GPU id]
