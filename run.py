import os
import torch
from src.utils import set_logger, parse_arguments
from src.train import train, evaluate_bleu, evaluate_loss_and_perplexity
from src.model import ModelConfigure, LSTM_Model, NMT
from src.data_utils import get_Jp_En_dataset, get_Jp_En_vocab, get_word2vec_model
from torch.utils.data.dataloader import DataLoader
from transformers import set_seed
from dataclasses import dataclass, field
from datetime import datetime
import wandb
from src.tokenizer import TokenizerDict, SimpleTokenizer, standardize_tokenizer

@dataclass
class DataArguments:
    dataset_path: str = field(
        default="data/eng_jpn.txt"
    )
    
    vocabulary_path: str = field(
        default="vocab/vocab.json"
    )
    
    src_word2vec_path: str = field(
        default="embedding/jp_embedding.bin"
    )
    
    tgt_word2vec_path: str = field(
        default="embedding/en_embedding.bin"
    )
    
    save_and_load_vocabulary: bool = field(
        default=True
    )
        
    pre_trained_model_path: str = field(
        default=None
    )

@dataclass
class TrainingArguments:
    lr: float = field(
        default=1e-3,
    )
    
    epoch: int = field(
        default=10
    )
    
    batch_size: int = field(
        default=32
    )
    
    save_dir: str = field(
        default="model_cache"
    )
    
    device: str = field(
        default="cuda",
        metadata={
            "choices": ["cuda", "cpu"]
        }
    )
    
    log_interval: int = field(
        default=10
    )
    
    seed: int = field(
        default=0
    )
    
    logging_level: str = field(
        default="INFO"
    )
    
    do_train: bool = field(
        default=False
    )
    
    do_eval: bool = field(
        default=False
    )
    
    use_wandb: bool = field(
        default=True
    )
    
def main():
    '''Parse arguments'''
    data_args, train_args, model_config = parse_arguments(DataArguments, TrainingArguments, ModelConfigure)

    '''Set seed'''
    set_seed(train_args.seed)
    
    '''Set logger'''
    set_logger(train_args.logging_level)
    
    '''Set wandb'''
    if train_args.use_wandb and train_args.do_train:
        wandb.login()
        wandb.init(project="RNN Translator", name="Debug", config=vars(train_args))
    
    '''Load dataset'''
    dataset = get_Jp_En_dataset(data_args.dataset_path)
    train_ds = dataset["train"]
    dev_ds = dataset["dev"]
    test_ds = dataset["test"]
        
    '''Get vocabulary'''
    vocab = get_Jp_En_vocab(dataset, data_args.vocabulary_path, save_and_load=data_args.save_and_load_vocabulary)
    
    '''Get pre-trained word2vec'''
    src_word2vec, tgt_word2vec = get_word2vec_model(data_args.src_word2vec_path, data_args.tgt_word2vec_path)
    # src_word2vec = tgt_word2vec = None
    
    '''Get tokenizer'''
    src_tokenizer = SimpleTokenizer(vocab.src, TokenizerDict["Japanese"], sep='', standardize=True)
    tgt_tokenizer = SimpleTokenizer(vocab.tgt, TokenizerDict["English"], sep=' ', standardize=True)
    
    '''Prepare data loader'''
    def collate_func(batch):
        src_sents = []
        tgt_sents = []
        for datum in batch:
            src_sents.append(datum['Japanese'])
            tgt_sents.append(datum['English'])
        src_padded, src_lengths, src_masks, sorted_indexes = src_tokenizer.encode(src_sents, to_tensor=True, pad=True, sort=True)
        tgt_sents = [tgt_sents[i] for i in sorted_indexes]
        tgt_padded, _, tgt_masks, _ = tgt_tokenizer.encode(tgt_sents, to_tensor=True, pad=True, sort=False)
        return src_padded, src_lengths, src_masks, tgt_padded, tgt_masks
    
    def bleu_collate_func(batch):
        src_sents = []
        tgt_sents = []
        for datum in batch:
            src_sents.append(datum['Japanese'])
            tgt_sents.append(datum['English'])
        src_sents, _, _, _ = src_tokenizer.encode(src_sents, to_tensor=False, pad=False, sort=False)
        tgt_sents, _, _, _ = tgt_tokenizer.encode(tgt_sents, to_tensor=False, pad=False, sort=False)
        return src_sents, tgt_sents
        
    
    train_loader = DataLoader(train_ds, batch_size=train_args.batch_size, shuffle=True, collate_fn=collate_func)
    dev_loader_perplexity = DataLoader(dev_ds, batch_size=1000, collate_fn=collate_func)
    dev_loader_bleu = DataLoader(dev_ds, batch_size=1000, collate_fn=bleu_collate_func)
    test_loader_perplexity = DataLoader(test_ds, batch_size=1000, collate_fn=collate_func)
    test_loader_bleu = DataLoader(test_ds, batch_size=1000, collate_fn=bleu_collate_func)
    
    '''Initialize model'''
    if train_args.do_train:
        model = LSTM_Model(model_config, vocab, src_word2vec, tgt_word2vec)
    elif train_args.do_eval and data_args.pre_trained_model_path is not None:
        model = LSTM_Model.load(data_args.pre_trained_model_path)

    '''Do training'''
    if train_args.do_train:
        train(train_loader, dev_loader_perplexity, dev_loader_bleu, model, train_args)
        file_name = f"{datetime.now().strftime("%m-%d-%H-%M")}_{train_args.epoch}_{train_args.lr}.bin"
        model.save(os.path.join(train_args.save_dir, file_name))
    
    if train_args.do_eval:
        model.to(train_args.device)
        nmt = NMT(model, src_tokenizer, tgt_tokenizer)
        out = nmt.translate_sentences(
            [
                "私の名前は愛です", # My name is love
                "昨日はお肉を食べません", # I didn't eat meat yesterday
                "いただきますよう", # I hope you enjoy it
                "秋は好きです", # I like autumn
                "おはようございます" # Good morning,
            ],
            beam_size=5
        )
        print(out)
        bleu = evaluate_bleu(test_loader_bleu, model, train_args)
        loss, perplexity = evaluate_loss_and_perplexity(test_loader_perplexity, model, train_args)
        print(bleu, loss, perplexity)
if __name__ == "__main__":
    main()