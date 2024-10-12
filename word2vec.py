from src.utils import set_logger, parse_arguments
from src.data_utils import get_Jp_En_dataset
from src.vocabulary import VocabEntry
from typing import List
import gensim
from dataclasses import dataclass, field
import logging
from gensim.test.utils import datapath
import os
from src.tokenizer import TokenizerDict, standardize_tokenizer

logger = logging.getLogger(__name__)

@dataclass
class Word2VecArgument:
    do_train: bool = field(
        default=False
    )
    
    language: str = field(
        default="English"
    )
    
    vector_size: int = field(
        default=256
    )

    min_count: int = field(
        default=5
    )
    
    window: int = field(
        default=5
    )
    
    workers: int = field(
        default=1
    )
    
    seed: int = field(
        default=0
    )
    
    epochs: int = field(
        default=5
    )
    
    sg: int = field(
        default=0
    )
    
    negative: int = field(
        default=5
    )
    
    dataset_path: str = field(
        default="data/eng_jpn.txt"
    )
    
    embedding_path: str = field(
        default="embedding/embedding.bin"
    )
    
    do_eval: bool = field(
        default=False
    )
    
    JWS_path: str = field(
        default="data/JWS"
    )


def train(args):
    dataset = get_Jp_En_dataset(args.dataset_path)["train"][args.language]
    tokenizer = standardize_tokenizer(TokenizerDict[args.language])
    dataset = [tokenizer(sent) for sent in dataset]            
    model = gensim.models.Word2Vec(
        sentences=dataset,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        seed=args.seed,
        workers=args.workers,
        epochs=args.epochs,
        sg=args.sg,
        negative=args.negative
    )
    logger.info(f"Latest training loss: {model.get_latest_training_loss()}")
    return model

def eval(model:gensim.models.Word2Vec, args):
    match args.language:
        case "Japanese":
            dataset_dir = args.JWS_path
            scores = {
                "adj": model.wv.evaluate_word_pairs(os.path.join(dataset_dir, "score_adj.tsv")),
                "adv": model.wv.evaluate_word_pairs(os.path.join(dataset_dir, "score_adv.tsv")),
                "noun": model.wv.evaluate_word_pairs(os.path.join(dataset_dir, "score_noun.tsv")),
                "verb": model.wv.evaluate_word_pairs(os.path.join(dataset_dir, "score_verb.tsv"))
            }
        case "English":
            scores = {
                "wordsim353": model.wv.evaluate_word_pairs(datapath("wordsim353.tsv")),
                "google-analogy": model.wv.evaluate_word_analogies(datapath("questions-words.txt"))[0]
            }
        case _:
            scores = None
            logger.error(f"Not implemented language: {args.language}")
    return scores

def main():
    set_logger(logging.INFO)
    args = parse_arguments(Word2VecArgument)[0]
    model = None
    if args.do_train:
        model = train(args)
        model.save(args.embedding_path)
    else:
        model = gensim.models.Word2Vec.load(args.embedding_path)
    assert model is not None
    logger.info(f"Model info: {model}")
    if args.do_eval:
        scores = eval(model, args)
        logger.info(f"Eval\n{scores}")

if __name__ == "__main__":
    main()