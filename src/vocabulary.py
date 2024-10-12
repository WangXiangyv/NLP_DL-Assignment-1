''' 
Adapted from Stanford CS224n (winter 2023) Assignment 3
'''
import os
from rich.progress import track
import logging
from typing import List
from collections import Counter
from .tokenizer import TokenizerDict, standardize_tokenizer
import json


logger = logging.getLogger(__name__)


class VocabEntry(object):
    """ Vocabulary Entry, i.e. structure containing either
    src or tgt language terms.
    """
    def __init__(self, word2id=None):
        """ Init VocabEntry Instance.
        @param word2id (dict): dictionary mapping words 2 indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<PAD>'] = 0   # Pad Token
            self.word2id['<UNK>'] = 1   # Unknown Token
            self.word2id['<SOS>'] = 2   # Start of Sentence Token
            self.word2id['<EOS>'] = 3   # End of Sentence Token
        self.pad_id = self.word2id['<PAD>']
        self.unk_id = self.word2id['<UNK>']
        self.sos_id = self.word2id['<SOS>']
        self.eos_id = self.word2id['<EOS>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word 
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained    
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError('vocabulary is read only')

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        """ Return mapping of index to word.
        @param wid (int): word index
        @returns word (str): word corresponding to index
        """
        return self.id2word[wid]

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
        @param word (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        @param sents (list[str] or list[list[str]]): sentence(s) in words
        @return word_ids (list[int] or list[list[int]]): sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sents (list[str]): list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]
    
    @staticmethod
    def from_corpus(corpus:List[str], language:str="English", size=None, freq_cutoff=None):
        tokenizer = standardize_tokenizer(TokenizerDict[language])
        vocab_entry = VocabEntry()
        words = [word for sent in track(corpus, description=f"Get {language} vocab from Corpus") for word in tokenizer(sent)]
        word_freq = Counter(words)
        valid_words = [w for w, v in word_freq.items() if freq_cutoff is None or v >= freq_cutoff]
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)
        if isinstance(size, int) and size > 0:
            top_k_words = top_k_words[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        logger.info(f"Successfully build {language} vocab from corpus. Size:{len(vocab_entry)}")
        return vocab_entry
    
    def save(self, path:os.PathLike):
        if os.path.exists(path):
            logger.warning(f"{path} already exists.")
        with open(path, "w") as fout:
            json.dump(self.word2id)
    @staticmethod
    def load(path:os.PathLike):
        if not os.path.exists(path):
            raise ValueError(f"{path} does not exist.")
        with open(path, "r") as fin:
            word2id = json.load(fin)
        return VocabEntry(word2id)
    
class BilingualVocab(object):
    """ Vocab encapsulating src and target langauges."""
    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        """ Init Vocab.
        @param src_vocab (VocabEntry): VocabEntry for source language
        @param tgt_vocab (VocabEntry): VocabEntry for target language
        """
        self.src = src_vocab
        self.tgt = tgt_vocab

    def save(self, file_path):
        """ Save Vocab to file as JSON dump.
        @param file_path (str): file path to vocab file
        """
        with open(file_path, 'w') as f:
            json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), f, indent=2)

    @staticmethod
    def load(file_path):
        """ Load vocabulary from JSON dump.
        @param file_path (str): file path to vocab file
        @returns Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, 'r'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']

        return BilingualVocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))
