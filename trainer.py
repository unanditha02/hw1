# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# Modified by Sudeep Dasari
# --------------------------------------------------------
from __future__ import print_function
from functools import total_ordering

import torch
import numpy as np

import utils
from voc_dataset import VOCDataset
from torch.utils.tensorboard import SummaryWriter

def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model):
    # TODO: Q2.2 Implement code for model saving
    filename = 'model_params/checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    torch.save(model.state_dict(), filename)


def train(args, model, optimizer, scheduler=None, model_name='model'):
    # TODO Q1.5: Initialize your tensorboard writer here!
    train_loader = utils.get_data_loader(
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    writer = SummaryWriter()
    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0
    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            # Get a batch of data
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculate the loss
            # TODO Q1.4: your loss for multi-label classification
            # criterion = torch.nn.MultiLabelSoftMarginLoss()
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(output, target)
            # Calculate gradient w.r.t the loss
            loss.backward()
            # Optimizer takes one step
            optimizer.step()
            # Log info
            if cnt % args.log_every == 0:
                # TODO Q1.5: Log training loss to tensorboard
                writer.add_scalar("Loss", loss.item(), cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                # TODO Q3.2: Log histogram of gradients
                for tag, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(tag, param.grad.data.cpu().numpy(), epoch)
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                # TODO Q1.5: Log MAP to tensorboard
                writer.add_scalar("mAP", map, cnt)
                model.train()
            cnt += 1

        # TODO Q3.2: Log Learning rate
        if scheduler is not None:
            scheduler.step()
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map
