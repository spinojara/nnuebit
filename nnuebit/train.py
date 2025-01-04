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

def loss_fn(output, eval, result, exponent, lam):
    wdl_output = torch.sigmoid(scaling * output * sigmoid_scaling)
    wdl_target = lam * torch.sigmoid(scaling * eval * sigmoid_scaling) + (1.0 - lam) * result
    return torch.sum(torch.pow(torch.abs(wdl_output - wdl_target), exponent))

def train(nnue, train_data, train_data_name, val_data, start_epoch, epochs, epoch_size, validation_size, device, lr, gamma, exponent, save_every, lam, weight_decay):
    epochs += start_epoch - 1

    start = time.time()

    decay = []
    no_decay = []

    for name, param in nnue.named_parameters():
        if name.endswith('.bias') or name == 'ft.weight':
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.Adam([
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
        ], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    batchbit.loader_reset(train_data)
    batchbit.loader_reset(val_data)

    for epoch in range(start_epoch, epochs + 1):
        t = time.time()
        print(f'starting epoch {epoch} of {epochs}')
        
        total = 0
        loss = 0
        cp = 0
        while total < validation_size:
            batch = batchbit.batch_fetch(val_data)
            if not batch:
                sys.exit(1)
            bucket, f1, f2, eval, result = batch.contents.get_tensors(device)
            total += batch.contents.size
            output = nnue(bucket, f1, f2)
            loss += loss_fn(output, eval, result, exponent, lam).item()
            cp += loss_fn(output, eval, result, 1.0, lam).item()
        loss /= total
        cp /= total

        losscp = 2 * inverse_sigmoid(1 / 2 + cp / 2)
        print(f'loss is {round(loss, 5)} ({round(losscp)} cp) for validation data')
        print('learning rate is now {:.2e}'.format(optimizer.param_groups[0]['lr']))
        
        total = 0
        while total < epoch_size:
            batch = batchbit.batch_fetch(train_data)
            if not batch:
                sys.exit(1)
            bucket, f1, f2, eval, result = batch.contents.get_tensors(device)
            total += batch.contents.size
            def closure():
                optimizer.zero_grad()
                output = nnue(bucket, f1, f2)
                loss = loss_fn(output, eval, result, exponent, lam) / batch.contents.size
                loss.backward()
                return loss
            nnue.clamp_weights()
            optimizer.step(closure)

        if save_every > 0 and epoch % save_every == 0 and epoch != epochs:
            name = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ.ckpt')
            save(name, nnue, train_data_name, epoch, lr, gamma, exponent, lam, weight_decay)


        scheduler.step()
        t = time.time() - t
        eta = time.time() + (epochs - epoch) * t
        print(f'epoch elapsed {round(t, 2)} seconds')
        print(f'estimated time of arrival is {time.strftime('%Y-%m-%d %H:%M', time.localtime(eta))}\n')

    print(f'training elapsed {round(time.time() - start, 2)} seconds')
    return optimizer.param_groups[0]['lr']

def save(name, nnue, train, epoch, lr, gamma, exponent, lam, weight_decay):
    print(f'epoch: {epoch}')
    print(f'train: {train}')
    print(f'lr: {lr}')
    print(f'gamma: {gamma}')
    print(f'exponent: {exponent}')
    print(f'lambda: {lam}')
    print(f'weight decay: {weight_decay}')
    torch.save({'nnue': nnue.state_dict(),
                'epoch': epoch,
                'train': train,
                'lr': lr,
                'gamma': gamma,
                'exponent': exponent,
                'lam': lam,
                'weight_decay': weight_decay}, name)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-data', type=str, help='Training data file', default=None)
    parser.add_argument('--validation-data', type=str, help='Validation data file', default=None)
    parser.add_argument('--random-skip', type=float, help='Random skipping frequency', default=0.8)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=16384)
    parser.add_argument('--device', type=str, help='Pytorch device', default='cuda')
    parser.add_argument('--load', type=str, help='Load pytorch checkpoint file (.ckpt)', default='')
    parser.add_argument('--info', action='store_true', help='Print information about the loaded checkpoint file')
    parser.add_argument('--override-training-data', action='store_true', help='Override the training data file used for checkpoint')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=400)
    parser.add_argument('--start-epoch', type=int, help='Starting epoch', default=1)
    parser.add_argument('--epoch-size', type=int, help='Number of positions per epoch', default=100000000)
    parser.add_argument('--validation-size', type=int, help='Number of positions for validation', default=1000000)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('--gamma', type=float, help='Scheduler gamma', default=0.992)
    parser.add_argument('--save-every', type=int, help='Save every <x> epochs', default=100)
    parser.add_argument('--exponent', type=float, help='Loss exponent', default=2.0)
    parser.add_argument('--lambda', dest='lam', type=float, help='Interpolate between evaluation and game results. 1.0 uses pure evaluation score, and 0.0 uses pure result as score.', default=1.0)
    parser.add_argument('--weight-decay', type=float, help='Weight decay', default=0.0)

    args, unknown = parser.parse_known_args()
    if len(unknown) >= 1:
        if args.training_data is None:
            args.training_data = unknown[0]
        elif args.validation_data is None:
            args.validation_data = unknown[0]
    if len(unknown) >= 2:
        if args.validation_data is None:
            args.validation_data = unknown[1]


    device = torch.device(args.device)

    nnue = model.nnue().to(device=device, non_blocking=True)

    if args.info and not args.load:
        print('--info without --load')
        sys.exit(1)

    if args.lam < 0.0 or args.lam > 1.0:
        print('lambda must be between 0.0 and 1.0')

    if args.load:
        if not args.load.endswith('.ckpt'):
            print('Loaded file is not a checkpoint file.')
            sys.exit(2)
        ckpt = torch.load(args.load)
        nnue.load_state_dict(ckpt['nnue'])
        args.start_epoch = ckpt['epoch'] + 1
        # Scale lr according to the scheduler
        args.lr = ckpt['lr'] * (ckpt['gamma'] ** ckpt['epoch'])
        args.gamma = ckpt['gamma']
        args.exponent = ckpt['exponent']
        args.lam = ckpt['lam']
        args.weight_decay = ckpt['weight_decay']
        if args.info:
            print(f'epochs: {ckpt['epoch']}')
            print(f'lr: {ckpt['lr']} ({args.lr})')
            print(f'gamma: {ckpt['gamma']}')
            print(f'exponent: {ckpt['exponent']}')
            print(f'data: {ckpt['train']}')
            print(f'lambda: {ckpt['lam']}')
            print(f'weight decay: {ckpt['weight_decay']}')
            return
        if args.training_data is not None and ckpt['train'] and ckpt['train'] != args.training_data:
            print(f'New training data file {args.training_data} is not the same as the old training data file {ckpt['train']}.')
            print('Pass the flag --override-training-data or use the old training data file.')
            return

    if batchbit.version() != model.VERSION_NNUE:
        print(f'version mismatch')
        sys.exit(1)

    if args.training_data is None:
        print('need --training-data')
    if args.validation_data is None:
        print('need --validation-data')
    if args.training_data is None or args.validation_data is None:
        sys.exit(1)

    train_data = batchbit.loader_open(args.training_data.encode(), args.batch_size, args.random_skip, args.lam < 1.0)
    if not train_data:
        sys.exit(1)
    val_data = batchbit.loader_open(args.validation_data.encode(), args.batch_size, 0.0, args.lam < 1.0)
    if not val_data:
        sys.exit(1)

    lr = train(nnue, train_data, args.training_data, val_data, args.start_epoch, args.epochs, args.epoch_size, args.validation_size, device, args.lr, args.gamma, args.exponent, args.save_every, args.lam, args.weight_decay)

    name = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ.ckpt')
    save(name, nnue, args.training_data, args.start_epoch - 1 + args.epochs, args.lr, args.gamma, args.exponent, args.lam, args.weight_decay)
    quantize.quantize(name)

    batchbit.loader_close(train_data)
    batchbit.loader_close(val_data)

if __name__ == '__main__':
    main()
