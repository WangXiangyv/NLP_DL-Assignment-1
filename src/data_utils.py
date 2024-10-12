from . import utils, vocabulary
from datasets import Dataset, DatasetDict, load_dataset
import os
import logging
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)

def get_Jp_En_dataset(path:os.PathLike, shuffle_seed=2024) -> DatasetDict:
    '''Load Chinese Japanese-English dataset'''
    ds = Dataset.from_csv(path, delimiter='\t')
    size = len(ds)
    dev_size = test_size = int(0.1*size)
    train_size = size - dev_size - test_size
    ds = ds.shuffle(seed=shuffle_seed)
    ds = DatasetDict({
        "train":ds.select(range(train_size)),
        "dev":ds.select(range(train_size, train_size+dev_size)),
        "test":ds.select(range(train_size+dev_size, size))
    })
    return ds
    

def get_Jp_En_vocab(dataset:DatasetDict, path:os.PathLike=None, save_and_load:bool=False):
    if path and os.path.exists(path) and save_and_load:
        logger.info(f"load existing bilingual vocabulary from {path}")
        vocab = vocabulary.BilingualVocab.load(path)
    else:
        src_vocab = vocabulary.VocabEntry.from_corpus(dataset['train']['Japanese'], language="Japanese")
        tgt_vocab = vocabulary.VocabEntry.from_corpus(dataset['train']['English'], language="English")
        vocab = vocabulary.BilingualVocab(src_vocab, tgt_vocab)
        if save_and_load:
            vocab.save(path)
    return vocab

def get_word2vec_model(src_path:os.PathLike, tgt_path:os.PathLike):
    return Word2Vec.load(src_path), Word2Vec.load(tgt_path)