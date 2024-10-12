import logging
import torch.utils
from .vocabulary import VocabEntry, BilingualVocab
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import List
from dataclasses import dataclass, field
from gensim.models import Word2Vec
from collections import namedtuple
from .tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

@dataclass
class ModelConfigure:
    embedding_dim: int = field(
        default=512
    )
    
    hidden_dim: int = field(
        default=512
    )
    
    static: bool = field(
        default=False
    )
    
    output_dropout_prob: float = field(
        default=0.1
    )
    
    embed_dropout_prob: float = field(
        default=0.1
    )

class BilingualEmbedding(nn.Module):
    def __init__(self, embedding_dim:int, vocab:BilingualVocab, src_word2vec:Word2Vec=None, tgt_word2vec:Word2Vec=None, static:bool=False) -> None:
        super(BilingualEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.static = static
        
        # Embedding layers
        self.src_embedding = self.create_embedding(self.embedding_dim, vocab.src, src_word2vec)
        self.tgt_embedding = self.create_embedding(self.embedding_dim, vocab.tgt, tgt_word2vec)
        
        # Set static embedding
        if static:
            self.src_embedding.requires_grad_(False)
            self.tgt_embedding.requires_grad_(False)
    
    @staticmethod
    def create_embedding(embedding_dim, vocab_entry:VocabEntry, word2vec:Word2Vec=None):
        if word2vec is None:
            embedding = nn.Embedding(len(vocab_entry), embedding_dim, padding_idx=vocab_entry.pad_id)
        else:
            assert word2vec.vector_size == embedding_dim
            embedding_matrix = torch.empty((len(vocab_entry), embedding_dim), dtype=torch.float32)
            nn.init.xavier_uniform_(embedding_matrix, gain=nn.init.calculate_gain("tanh"))
            for word, idx in vocab_entry.word2id.items():
                if word in word2vec.wv:
                    embedding_matrix[idx] = torch.tensor(word2vec.wv[word], dtype=torch.float32)
            embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=vocab_entry.pad_id, freeze=False)
        return embedding
                    
class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super(BahdanauAttention, self).__init__()
        self.W = nn.Linear(enc_dim, dec_dim)
        self.U = nn.Linear(dec_dim, dec_dim)
        self.V = nn.Linear(dec_dim, 1)
    def forward(self, enc_hiddens, dec_hidden):
        '''
        enc_hiddens: (B, L, enc_H)
        dec_hidden: (B, dec_H)
        '''
        return self.V(F.tanh(self.W(enc_hiddens) + self.U(dec_hidden).unsqueeze(dim=1)))

class LSTM_Model(nn.Module):
    '''
    Simple RNN machine translator model.
    Encoder: bidirectional LSTM
    Decoder: unidirectional LSTM with attention mechanism
    '''
    def __init__(self, config, vocab:BilingualVocab=None, src_word2vec:Word2Vec=None, tgt_word2vec:Word2Vec=None) -> None:
        super(LSTM_Model, self).__init__()
        self.config = config
        self.vocab = vocab
        self.src_vocab_size = len(vocab.src)
        self.tgt_vocab_size = len(vocab.tgt)
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.output_dropout_prob = config.output_dropout_prob
        self.embed_dropout_prob = config.embed_dropout_prob
        self.enc_ln = nn.LayerNorm(2*self.hidden_dim)
        self.dec_ln = nn.LayerNorm(self.hidden_dim)

        # Embedding layer
        self.embeddings = BilingualEmbedding(self.embedding_dim, self.vocab, src_word2vec, tgt_word2vec)
        
        # Convolution layer
        self.convolution = nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=2, padding="same")
        
        # Encoder layer
        self.encoder = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True)
        
        # Decoder cell
        self.decoder = nn.LSTMCell(self.embedding_dim+self.hidden_dim, self.hidden_dim)
        
        # Linear projection layers
        self.h_proj = nn.Linear(2*self.hidden_dim, self.hidden_dim) # encoder last h to decoder first h
        self.c_proj = nn.Linear(2*self.hidden_dim, self.hidden_dim) # encoder last c to decoder first c
        self.att_proj = nn.Linear(2*self.hidden_dim, self.hidden_dim, bias=False) # attention proj
        self.out_proj = nn.Linear(3*self.hidden_dim, self.hidden_dim) # output proj
        self.vocab_proj = nn.Linear(self.hidden_dim, self.tgt_vocab_size) # output_dim to vocab_size
        self.bahdanau_att = BahdanauAttention(2*self.hidden_dim, self.hidden_dim)
        
        # Dropout layer
        self.enc_embed_dropout = nn.Dropout(self.embed_dropout_prob)
        self.dec_embed_dropout = nn.Dropout(self.embed_dropout_prob)
        self.output_dropout = nn.Dropout(self.output_dropout_prob)
        
    def forward(self, src_padded:torch.Tensor, src_lengths:List[int], src_masks:torch.Tensor, tgt_padded:torch.Tensor, tgt_masks:torch.Tensor):
        '''
        src_padded (src_len, B)
        src_masks (src_len, B)
        tgt_padded (tgt_len, B)
        tgt_masks (tgt_len, B)
        ps: src_padded and src_lengths should be sorted from long to short
        '''
        src_masks = src_masks.transpose(0, 1) # (B, src_len)
        
        # encode
        enc_hiddens, final_hidden, final_cell = self.encode(src_padded, src_lengths)
        enc_hiddens = self.enc_ln(enc_hiddens)
        
        # calculate init decoder state
        init_dec_hidden = self.h_proj(final_hidden) # (B, H)
        init_dec_cell = self.c_proj(final_cell) # (B, H)
        
        # decode
        dec_outputs = self.decode(enc_hiddens, src_masks, init_dec_hidden, init_dec_cell, tgt_padded) # (tgt_len-1, B, H)
        
        # softmax
        P = F.log_softmax(self.vocab_proj(dec_outputs), dim=-1) # (tgt_len-1, B, vocab)

        # negative log likelihood loss
        log_likelihood = torch.gather(P, index=tgt_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * tgt_masks[1:] # (tgt_len-1, B)
        loss = -log_likelihood.sum(dim=0) # (B,)
        
        return loss
    
    def encode(self, src_padded:torch.Tensor, src_lengths:List[int]):
        '''
        src_padded (L, B)
        '''
        # embed
        src_padded = self.embeddings.src_embedding(src_padded) # (L, B, E)
        # convolve
        src_padded = F.gelu(self.convolution(src_padded.permute((1, 2, 0))).permute(2, 0, 1)) + src_padded # (L, B, E)
        # embed dropout
        src_padded = self.enc_embed_dropout(src_padded)
        # feed into LSTM
        enc_hiddens, (final_hidden, final_cell) = self.encoder(pack_padded_sequence(src_padded, src_lengths)) # enc_hiddens PackSequence; final_hidden/cell (2, B, H)
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens, padding_value=self.vocab.src.pad_id) # (L, B, 2*H)
        enc_hiddens = enc_hiddens.swapaxes(0, 1) # (B, L, 2*H)
        final_hidden = torch.cat((final_hidden[0], final_hidden[1]), dim=1) # (B, 2*H)
        final_cell = torch.cat((final_cell[0], final_cell[1]), dim=1) # (B, 2*H)
        
        return enc_hiddens, final_hidden, final_cell
    
    def decode(self, enc_hiddens, src_masks, init_dec_hidden, init_dec_cell, tgt_padded):
        '''
        enc_hiddens (B, src_len, 2*H)
        src_masks (B, src_len)
        init_dec_hidden/cell (B, H)
        tgt_padded (tgt_len, B)
        '''
        # discard '<EOS>' token of max len sentences
        tgt_padded = tgt_padded[:-1]
        
        # embed
        tgt_padded = self.embeddings.tgt_embedding(tgt_padded) # (tgt_len-1, B, E)
        
        # attention projection
        enc_hiddens_proj = self.att_proj(enc_hiddens) # (B, src_len, H)
        
        # initial output and state
        output_t = torch.zeros((enc_hiddens.shape[0], self.hidden_dim), device=self.device) # (B, H)
        dec_hidden_t, dec_cell_t = init_dec_hidden, init_dec_cell
        
        outputs = []
        # recurrent
        for Y in torch.split(tgt_padded, split_size_or_sections=1, dim=0): # Y (1, B, E)
            Y = Y.squeeze() # (B, E)
            input_t = torch.cat((output_t, Y), dim=1) # (B, H+E)
            dec_hidden_t, dec_cell_t, output_t = self.decode_step(input_t, dec_hidden_t, dec_cell_t, enc_hiddens_proj, enc_hiddens, src_masks)
            outputs.append(output_t)

        dec_outputs = torch.stack(outputs) # (tgt_len-1, B, H)
        return dec_outputs
    
    def decode_step(self, input, last_dec_hidden, last_dec_cell, enc_hiddens_proj, enc_hiddens, src_masks):
        '''
        input (B, H+E)
        last_dec_hidden/cell (B, H)
        enc_hiddens_proj (B, src_len, H)
        enc_hiddens (B, src_len, H)
        enc_masks (B, src_len)
        '''
        dec_hidden, dec_cell = self.decoder(input, (last_dec_hidden, last_dec_cell)) # (B, H), (B, H)
        # alpha = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(dim=2)).squeeze(dim=2) # (B, src_len)
        alpha = self.bahdanau_att(enc_hiddens, dec_hidden).squeeze(dim=2)
        if src_masks is not None:
            alpha.masked_fill_(~src_masks.bool(), -float('inf')) # (B, src_len)
        alpha = F.softmax(alpha, dim=-1) # (B, src_len)
        a = torch.bmm(alpha.unsqueeze(dim=1), enc_hiddens).squeeze(dim=1) # (B, 2*H)
        u = torch.cat((a, self.dec_ln(dec_hidden)), dim=1) # (B, 3*H)
        output = self.output_dropout(F.tanh(self.out_proj(u))) # (B, H)
        
        return dec_hidden, dec_cell, output
    

    def beam_search(self, src_sentence:torch.Tensor, beam_size:int=5, max_decoding_time_step:int=64) -> List[Hypothesis]:
        '''
        src_sentence (L, 1)
        '''
        # encode
        enc_hiddens, final_hidden, final_cell = self.encode(src_sentence, [src_sentence.size(0)]) # (1, L, 2*H) (1, 2*H) (1, 2*H)
        enc_hiddens = self.enc_ln(enc_hiddens)
        
        # calculate initial decoder state
        dec_hidden_t = self.h_proj(final_hidden) # (1, H)
        dec_cell_t = self.c_proj(final_cell) # (1, H)
        output_t = torch.zeros(1, self.hidden_dim, device=self.device) # (1, H)
        
        # attention projection
        enc_hiddens_proj = self.att_proj(enc_hiddens) # (1, L, H)
        
        hypotheses = [[self.vocab.tgt.sos_id]]
        hyp_scores = torch.zeros(1, dtype=torch.float32, device=self.device)
        completed_hypotheses = []
        
        T = 0
        while len(completed_hypotheses) < beam_size and T < max_decoding_time_step:
            
            T += 1
            
            # Pretend to have batch size of hyp_num
            hyp_num = len(hypotheses)
            expanded_enc_hiddens = enc_hiddens.expand(hyp_num, enc_hiddens.size(1), enc_hiddens.size(2)) # (hyp, L, 2*H)
            expanded_enc_hiddens_proj = enc_hiddens_proj.expand(hyp_num, enc_hiddens_proj.size(1), enc_hiddens_proj.size(2)) # (hyp, L, H)
            Y_t = torch.tensor([hyp[-1] for hyp in hypotheses], dtype=torch.long, device=self.device) # (hyp,)
            Y_t = self.embeddings.tgt_embedding(Y_t) # (hyp, E)
            input_t = torch.cat((output_t, Y_t), dim=1)
            
            # decode step
            dec_hidden_t, dec_cell_t, output_t = self.decode_step(input_t, dec_hidden_t, dec_cell_t, expanded_enc_hiddens_proj, expanded_enc_hiddens, src_masks=None)
            
            # Update beam
            log_P_t = F.log_softmax(self.vocab_proj(output_t)) # (hyp, vocab)
            available_num = beam_size - len(completed_hypotheses)
            

            scores_t = torch.flatten(log_P_t + hyp_scores.unsqueeze(1).expand_as(log_P_t))
            top_scores, top_poses = torch.topk(scores_t, k=available_num)
            
            top_hyp_ids = top_poses // len(self.vocab.tgt)
            top_vocab_ids = top_poses % len(self.vocab.tgt)
            
            new_hypotheses = []
            new_scores = []
            corresponding_prev_hyp_ids = []
            for score, hyp_id, vocab_id in zip(top_scores, top_hyp_ids, top_vocab_ids):
                score, hyp_id, vocab_id = score.item(), hyp_id.item(), vocab_id.item()
                if vocab_id == self.vocab.tgt.eos_id:
                    completed_hypotheses.append(Hypothesis(hypotheses[hyp_id] + [vocab_id], score))
                else:
                    new_scores.append(score)
                    new_hypotheses.append(hypotheses[hyp_id] + [vocab_id])
                    corresponding_prev_hyp_ids.append(hyp_id)
            
            # Update next step data
            if len(completed_hypotheses) == beam_size:
                break
            corresponding_prev_hyp_ids = torch.tensor(corresponding_prev_hyp_ids, dtype=torch.long, device=self.device)
            dec_hidden_t = dec_hidden_t[corresponding_prev_hyp_ids]
            dec_cell_t = dec_cell_t[corresponding_prev_hyp_ids]
            output_t = output_t[corresponding_prev_hyp_ids]
            
            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_scores, dtype=torch.float32, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0], score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score/len(hyp.value), reverse=True)
        return completed_hypotheses
        
    @property
    def device(self) -> torch.device:
        return self.embeddings.src_embedding.weight.device
    
    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path)
        model = LSTM_Model(params['config'], params['vocab'])
        model.load_state_dict(params['state_dict'])
        return model

    def save(self, path: str):
        logger.info(f"save model parameters to {path}")
        params = {
            'config': self.config,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

class NMT:
    '''
    A simple wrapper for interact with LSTM model with raw string
    '''
    def __init__(self, model:LSTM_Model, src_tokenzier:SimpleTokenizer, tgt_tokenizer:SimpleTokenizer) -> None:
        self.model = model
        self.src_tokenizer = src_tokenzier
        self.tgt_tokenizer = tgt_tokenizer
    
    def translate_sentences(self, src_sents:str|List[str], beam_size=5):
        if isinstance(src_sents, str):
            src_sents = [src_sents]
        predictions = []
        for sent in src_sents:
            self.model.eval()
            sent_tensor, _, _  = self.src_tokenizer.encode(sent, to_tensor=True)
            sent_tensor = sent_tensor.to(self.model.device)
            hypotheses = self.model.beam_search(sent_tensor, beam_size=beam_size)
            # predictions.append([self.tgt_tokenizer.decode(hyp.value) for hyp in hypotheses])
            predictions.append(self.tgt_tokenizer.decode(hypotheses[0].value))
        if len(predictions) == 1:
            predictions = predictions[0]
        return predictions
    
    