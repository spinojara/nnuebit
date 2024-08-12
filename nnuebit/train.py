#!/usr/bin/env python3

import torch
import ctypes
import time
import math
import argparse
import datetime
import sys

from . import model
from . import batchbit
from . import quantize

sigmoid_scaling = math.log(10) / 400
scaling = (127 * 64 / 16)
def sigmoid(x):
    return 1 / (1 + math.exp(-sigmoid_scaling * x))

def inverse_sigmoid(y):
    return -math.log(1 / y - 1) / sigmoid_scaling

def loss_fn(output, target, exponent):
    wdl_output = torch.sigmoid(scaling * output * sigmoid_scaling)
    wdl_target = torch.sigmoid(scaling * target * sigmoid_scaling)
    return torch.sum(torch.pow(torch.abs(wdl_output - wdl_target), exponent))

def train(nnue, train_data, val_data, start_epoch, epochs, device, lr, gamma, exponent, save_every):
    start = time.time()

    optimizer = torch.optim.Adam(nnue.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(start_epoch, epochs + start_epoch):
        t = time.time()
        print(f'starting epoch {epoch} of {epochs}')
        
        batchbit.batch_reset(val_data)
        loss = 0
        total = 0
        cp = 0
        while True:
            batch = batchbit.next_batch(val_data)
            if (batch.contents.is_empty()):
                break
            f1, f2, target = batch.contents.get_tensors(device)
            total += batch.contents.actual_size
            output = nnue(f1, f2)
            loss += loss_fn(output, target, exponent).item()
            cp += loss_fn(output, target, 1.0).item()
        loss /= total
        cp /= total

        losscp = 2 * inverse_sigmoid(1 / 2 + cp / 2)
        print(f'loss is {round(loss, 5)} ({round(losscp)} cp) for validation data')
        print('learning rate is now {:.2e}'.format(optimizer.param_groups[0]['lr']))
        
        batchbit.batch_reset(train_data)
        while True:
            batch = batchbit.next_batch(train_data)
            if (batch.contents.is_empty()):
                break
            f1, f2, target = batch.contents.get_tensors(device)
            def closure():
                optimizer.zero_grad()
                output = nnue(f1, f2)
                loss = loss_fn(output, target, exponent) / batch.contents.actual_size
                loss.backward()
                return loss
            nnue.clamp_weights()
            optimizer.step(closure)

        if save_every > 0 and epoch % save_every == 0:
            save('temp.ckpt', nnue, '', epoch, lr, gamma, exponent)


        scheduler.step()
        t = time.time() - t
        eta = time.time() + (epochs - epoch) * t
        print(f'epoch elapsed {round(t, 2)} seconds')
        print(f'estimated time of arrival is {time.strftime('%Y-%m-%d %H:%M', time.localtime(eta))}\n')

    print(f'training elapsed {round(time.time() - start, 2)} seconds')
    return optimizer.param_groups[0]['lr']

def save(name, nnue, train, epoch, lr, gamma, exponent):
    print(f'epoch: {epoch}')
    print(f'train: {train}')
    print(f'lr: {lr}')
    print(f'gamma: {gamma}')
    print(f'exponent: {exponent}')
    torch.save({'nnue': nnue.state_dict(),
                'epoch': epoch,
                'train': train,
                'lr': lr,
                'gamma': gamma,
                'exponent': exponent}, name)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('training_data', type=str, help='Training data file')
    parser.add_argument('validation_data', type=str, help='Validation data file')
    parser.add_argument('--random-skip', type=float, help='Random skipping frequency', default=0.8)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=16384)
    parser.add_argument('--device', type=str, help='Pytorch device', default='cuda')
    parser.add_argument('--load', type=str, help='Load pytorch checkpoint file (.ckpt)', default='')
    parser.add_argument('--info', action='store_true', help='Print information about the loaded checkpoint file')
    parser.add_argument('--override-training-data', action='store_true', help='Override the training data file used for checkpoint')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=400)
    parser.add_argument('--start-epoch', type=int, help='Starting epoch', default=1)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('--gamma', type=float, help='Scheduler gamma', default=0.992)
    parser.add_argument('--save-every', type=int, help='Save every <x> epochs', default=20)
    parser.add_argument('--exponent', type=float, help='Loss exponent', default=2.0)

    args = parser.parse_args()

    nnue = model.nnue().to(device=args.device, non_blocking=True)

    if args.info and not args.load:
        print('--info without --load')
        sys.exit(1)

    if args.load:
        ckpt = torch.load(args.load)
        nnue.load_state_dict(ckpt['nnue'])
        args.start_epoch = ckpt['epoch'] + 1
        # Scale lr according to the scheduler
        args.lr = ckpt['lr'] * (ckpt['gamma'] ** ckpt['epoch'])
        args.gamma = ckpt['gamma']
        args.exponent = ckpt['exponent']
        if args.info:
            print(f'epochs: {ckpt['epoch']}')
            print(f'lr: {ckpt['lr']} ({args.lr})')
            print(f'gamma: {ckpt['gamma']}')
            print(f'exponent: {ckpt['exponent']}')
            return
        if ckpt['train'] and ckpt['train'] != args.training_data:
            print(f'New training data file {args.training_data} is not the same as the old training data file {ckpt['train']}.')
            print('Pass the flag --override-training-data or use the old training data file.')
            return

    train_data = batchbit.batch_open(args.training_data.encode(), args.batch_size, args.random_skip)
    val_data = batchbit.batch_open(args.validation_data.encode(), args.batch_size, 0.0)

    lr = train(nnue, train_data, val_data, args.start_epoch, args.epochs, args.device, args.lr, args.gamma, args.exponent, args.save_every)

    name = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ.ckpt')
    train.save(name, nnue, args.training_data, args.start_epoch - 1 + args.epochs, args.lr, args.gamma, args.exponent)
    quantize.quantize(name)

    batchbit.batch_close(train_data)
    batchbit.batch_close(val_data)

if __name__ == '__main__':
    main()
