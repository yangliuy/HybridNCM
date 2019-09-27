# Hybrid-Neural-Conversation-Model (HybridNCM)
Code on A Hybrid Retrieval-Generation Neural Conversation Model (CIKM 2019).

## Introduction

This repository maintains the implementation of the hybrid retrieval-generation neural conversation models proposed in the CIKM 2019 paper "A Hybrid Retrieval-Generation Neural Conversation Model".

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

## Organization of the Code Repository
### Data Set
We can not open source the Twitter/ Foursqure data used in our CIKM 2019 paper since it is an internal data set in Microsoft Research. To demo how to train and test the model, we provide demo data generated based on the [Ubuntu Dialog Corpus](https://arxiv.org/abs/1506.08909) data in the folder HybridNCM/demo_data. You can easily adapt these instructions to other data sets in your lab or company. The demo data set can be downloaded from this [Google Drive folder](https://drive.google.com/drive/folders/14kp-q1nre-mKjO4ExfXAVyhuGK3woxsq?usp=sharing).

### Code for the Generation Module
The code for the generation module can be found in the folder HybridNCM/generation. It contains the implementation of the Seq2Seq model and Seq2Seq-Facts with Pytorch for response generation. Please check the readme file for this module for the details.

### Code for the Retrieval Module
The code for the retrieval module can be found in the folder HybridNCM/retrieval. The implementation of building search index and BM25 retrieval is based on [Lucene](https://lucene.apache.org/).

### Code for the Hybrid Ranking Module
The code for the hybrid ranking module can be found in the folder HybridNCM/hybrid_ranking. The implementation of the neural ranking model is based on [MatchZoo](https://github.com/NTMC-Community/MatchZoo). Please check the readme file for this module for the details.

## Contact information
For help or issues on running the code, please submit a GitHub issue.

For personal communication related to HybridNCM, please contact Liu Yang (yangliuyx@gmail.com) for questions about the retrieval module and hybrid ranking model, or Junjie Hu (junjieh@cs.cmu.edu) for questions about the generation module.
