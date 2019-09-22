## Introduction

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

Generation Module
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

Run the following command to generate the test sentences.

    bash test.sh [GPU id]
