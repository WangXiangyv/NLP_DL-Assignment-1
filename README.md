# LSTM based Jpn-Eng NMT

## Simple introduction of the project files

- Files in _data_ directory are Jpn-Eng dataset and Japanese Word Similarity (JWS) dataset.

- Files in _embedding_ directory are pre-trained word2vec model.

- Files in _src_ directory are the main implements of word2vec model and NMT model, coupled with auxiliary modules (e.g., functions for loading different datasets).

- Files in _vocab_ directory are pre-produced vocabulary, deriving from the training set.

- Other files in the root directory of the project are scripts.

## Instructions on run the model

### 1. Create environment

- Install packages by _requirements.txt_
- Run _download.py_ to prepare necessary resources

### 2. Run NMT model

Just run _run.sh_ to launch the model for training or for evaluation. You can try different parameters in _run.sh_. If you want to explore more adjustable parameters, just scan arguments parsing part in _run.py_ and adapt _run.py_