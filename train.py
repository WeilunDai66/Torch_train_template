import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import torch
import torch.nn as nn
import numpy as np

from utils import seed_everything, CosineAnnealingWarmupLR

def train(cfg):

    seed_everything()

    model

    train_dataset = _Dataset(cfg, 'train')
    train_loader = DataLoader(
            train_dataset,
            cfg.batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=True,
            drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingWarmupLR(optimizer, total_iters=cfg.epochs*len(train_loader), warmup_iters=0.2*cfg.epochs*len(train_loader))

    writer = SummaryWriter(cfg.TRAIN_WRITER)


    for epoch in range(cfg.epochs):
        progress_bar = tqdm(range(0, len(train_loader)))
        
        progress_bar.set_description(f'Epoch {epoch + 1}/{cfg.epochs}')

        model.train()

        for step, () in enumerate(train_loader):
            progress_bar.update(1)
            logs = {"step_loss": loss.item(), 'trans': t_loss.item(), 'rot': r_loss.detach().item()}
            progress_bar.set_postfix(**logs)

            writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch*len(train_loader) + step)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()

        model.eval()
        progress_bar = tqdm(range(0, len(test_loader)))
        progress_bar.set_description(f'Epoch {epoch}/{cfg.epochs}')



        with torch.no_grad():
            for step, () in enumerate(test_loader):

                progress_bar.update(1)





if __name__ == '__main__':
    from config import cfg

    if not os.path.exists(cfg.output_checkpoint):
        os.makedirs(cfg.output_checkpoint)

    train(cfg)