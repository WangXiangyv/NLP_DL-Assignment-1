import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import wandb
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import numpy as np
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def train(train_ds_loader:DataLoader, dev_ds_loader_perplexity:DataLoader, dev_ds_loader_bleu:DataLoader, model:nn.Module, args):
    if not torch.cuda.is_available() and "cuda" in args.device:
        raise ValueError(f"Device: {args.device} is not available!")
    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    os.mkdir(save_dir)

    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    # scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=22)
    # scheduler_2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=(scheduler_1, scheduler_2), milestones=[7])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    step = 0
    best_loss = 1000
    best_bleu = 0
    best_perplexity = 1000
    for epoch in range(1, args.epoch+1):
        for src_padded, src_lengths, src_masks, tgt_padded, tgt_masks in train_ds_loader:
            model.train()
            src_padded = src_padded.to(args.device)
            src_masks = src_masks.to(args.device)
            tgt_padded = tgt_padded.to(args.device)
            tgt_masks = tgt_masks.to(args.device)

            optimizer.zero_grad()
            loss = model(src_padded, src_lengths, src_masks, tgt_padded, tgt_masks) # (B,)
            loss = loss.sum()/args.batch_size
            loss.backward()
            optimizer.step()
            
            step += 1
            if args.use_wandb:
                wandb.log({"batch":step, "loss":loss.item()})   
            if step % args.log_interval == 0:
                logger.info(f"Epoch: {epoch} - Batch: {step} - Loss: {loss.item()}")
        scheduler.step()
        loss_score, perplexity_score = evaluate_loss_and_perplexity(dev_ds_loader_perplexity, model, args)
        bleu_score = evaluate_bleu(dev_ds_loader_bleu, model, args)
        logger.info(
            f"[yellow]Epoch[/yellow]: {epoch} - [yellow]eval_loss[/yellow]: {loss_score} - [yellow]Perplexity[/yellow]: {perplexity_score} - [yellow]BLEU[/yellow]: {bleu_score}"
        )
        if loss_score < best_loss:
            best_loss = loss_score
            model.save(os.path.join(save_dir, "best_loss_model.pth"))
        if best_bleu < bleu_score:
            best_bleu = bleu_score
            model.save(os.path.join(save_dir, "best_bleu_model.pth"))
        if perplexity_score < best_perplexity:
            best_perplexity = perplexity_score
            model.save(os.path.join(save_dir, "best_perplexity_model.pth"))
        if args.use_wandb:
            wandb.log(
                {
                    "epoch":epoch,
                    "eval_loss": loss_score,
                    "bleu":bleu_score, 
                    "perlexity":perplexity_score,
                    "best_eval_loss": best_loss,
                    "best_bleu":best_bleu, 
                    "best_perplexity":best_perplexity
                }
            )
    model.save(os.path.join(save_dir, "final_model.pth"))


def evaluate_loss_and_perplexity(ds_loader:DataLoader, model:nn.Module, args):
    tot_loss = 0
    tot_words = 0
    for src_padded, src_lengths, src_masks, tgt_padded, tgt_masks in ds_loader:
        model.eval()
        src_padded = src_padded.to(args.device)
        src_masks = src_masks.to(args.device)
        tgt_padded = tgt_padded.to(args.device)
        tgt_masks = tgt_masks.to(args.device)
        loss = model(src_padded, src_lengths, src_masks, tgt_padded, tgt_masks) # (B,)
        tot_loss += loss.sum().item()
        tot_words += sum(src_lengths) - len(src_lengths)
    avg_loss = tot_loss/len(ds_loader.dataset)
    perplexity = np.exp(tot_loss/tot_words)
    return avg_loss, perplexity


def evaluate_bleu(ds_loader:DataLoader, model:nn.Module, args):
    model.eval()
    bleu = 0
    for src_sents, tgt_sents in ds_loader:
        for src, tgt in zip(src_sents, tgt_sents):
            hypothesis = model.beam_search(torch.tensor(src, device=args.device).unsqueeze(1))[0].value
            bleu += sentence_bleu(references=[tgt], hypothesis=hypothesis)
    bleu = 100 * bleu / len(ds_loader.dataset)
    return bleu