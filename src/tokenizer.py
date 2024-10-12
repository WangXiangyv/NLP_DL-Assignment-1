import torch.utils
import torch
from typing import List, Callable
import jieba # Chinese tokenizer
import fugashi # Japanese tokenizer
import unidic # Japanese dictionary
from nltk.tokenize import word_tokenize # English tokenizer

TokenizerDict = {
    "English":word_tokenize,
    "Japanese":fugashi.Tagger(unidic.DICDIR),
    "Chinese":jieba.lcut
}

def standardize_tokenizer(tokenizer):
    '''A decorator for standardizing tokenizer function'''
    def standardized_tokenizer(sent, *args, **kwargs):
        tokenized_sent = tokenizer(sent.lower(), *args, **kwargs)
        ret = [str(word) for word in tokenized_sent]
        return ret
    return standardized_tokenizer

class SimpleTokenizer:
    def __init__(self, vocab, tokenizer:Callable, sep:str = ' ', standardize:bool=False):
        self.vocab = vocab
        self.tokenizer = standardize_tokenizer(tokenizer) if standardize else tokenizer
        self.sep = sep
    
    def encode(self, sents:str|List[str], to_tensor:bool=False, pad=False, batch_first=False, sort=False):
        if isinstance(sents, list):
            return self._encode_sentences(sents, to_tensor, pad, batch_first, sort)
        else:
            return self._encode_sentence(sents, to_tensor, batch_first)
    
    def decode(self, sents, lengths=None, batch_first=False):
        if lengths is not None:
            return self._decode_sentences(sents, lengths, batch_first)
        else:
            return self._decode_sentence(sents, lengths, batch_first)
    
    def _encode_sentences(self, sents:List[str], to_tensor:bool=False, pad=False, batch_first=False, sort=False):
        if to_tensor: assert pad
        lengths = masks = sorted_indexes = None
        cut_sents = [[self.vocab.sos_id]+self.vocab.words2indices(self.tokenizer(sent))+[self.vocab.eos_id] for sent in sents]
        if sort:
            sorted_indexes = range(len(cut_sents))
            sorted_indexes = sorted(sorted_indexes, key=lambda i:len(cut_sents[i]), reverse=True)
            cut_sents.sort(key=len, reverse=True)
        if pad:
            lengths = []
            masks = []
            if sort:
                max_len = len(cut_sents[0])
            else:
                max_len = len(max(cut_sents, key=len))
            for s in cut_sents:
                lengths.append(len(s))
                masks.append(len(s)*[1] + (max_len - len(s))*[0])
                if len(s) < max_len:
                    s.extend((max_len - len(s))*[self.vocab.pad_id])
        if to_tensor:
            cut_sents = torch.LongTensor(cut_sents)
            masks = torch.tensor(masks)
            if not batch_first:
                cut_sents = cut_sents.T
                masks  = masks.T

        return cut_sents, lengths, masks, sorted_indexes

    def _encode_sentence(self, sent:str, to_tensor:bool=False, pad=False, batch_first=False):
        sent = [self.vocab.sos_id]+self.vocab.words2indices(self.tokenizer(sent))+[self.vocab.eos_id]
        length = len(sent)
        mask = length*[1]
        if to_tensor:
            sent = torch.LongTensor(sent)
            mask = torch.tensor(mask)
            if not batch_first:
                sent = sent.unsqueeze(1)
        return sent, length, mask
    
    def _decode_sentences(self, sents_padded, lengths, batch_first=False):
        if isinstance(sents_padded, torch.Tensor) and not batch_first:
            sents_padded = sents_padded.T
        return [self.sep.join(self.vocab.indices2words(sent[1:l-1])) for sent,l in zip(sents_padded, lengths)]
    
    def _decode_sentence(self, sent, length, batch_first=False):
        if isinstance(sent, torch.Tensor) and not batch_first:
            sent = sent.squeeze(1)
        return self.sep.join(self.vocab.indices2words(sent[1:-1]))