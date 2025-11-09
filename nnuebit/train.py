#!/usr/bin/env python3

import torch
import ctypes
import time
import math
import argparse
import sys
from typing import Any

from . import model
from . import batchbit
from . import customuuid

sigmoid_scaling = math.log(10) / 400
scaling = (127 * 64 / 16)
def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-sigmoid_scaling * x))

def inverse_sigmoid(y: float) -> float:
    return -math.log(1 / y - 1) / sigmoid_scaling

def loss_fn(output: torch.Tensor, eval: torch.Tensor, result: torch.Tensor, exponent: float, lam: float) -> torch.Tensor:
    wdl_output = torch.sigmoid(scaling * output * sigmoid_scaling)
    wdl_target = lam * torch.sigmoid(scaling * eval * sigmoid_scaling) + (1.0 - lam) * result
    return torch.sum(torch.pow(torch.abs(wdl_output - wdl_target), exponent))

def train(nnue: model.NNUE, train_data: ctypes.c_void_p, val_data: ctypes.c_void_p, start_epoch: int, epochs: int, epoch_size: int, validation_size: int, device: torch.device, lr: float, gamma: float, exponent: float, save_every: int, lam: float, weight_decay: float, uuid: customuuid.UUID8, filename: str):
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
        print('starting epoch %d of %d' % (epoch, epochs))

        total = 0
        loss: float = 0
        cp: float = 0
        while total < validation_size:
            batch = batchbit.batch_fetch(val_data)
            if not batch:
                sys.exit(1)
            f1, f2, eval, result = batch.contents.get_tensors(device)
            total += batch.contents.size
            output = nnue(f1, f2)
            loss += loss_fn(output, eval, result, exponent, lam).item()
            cp += loss_fn(output, eval, result, 1.0, lam).item()
        loss /= total
        cp /= total

        losscp = 2 * inverse_sigmoid(1 / 2 + cp / 2)
        print('loss is %.5f (%d cp) for validation data' % (loss, round(losscp)))
        print('learning rate is now {:.2e}'.format(optimizer.param_groups[0]['lr']))

        total = 0
        while total < epoch_size:
            batch = batchbit.batch_fetch(train_data)
            if not batch:
                sys.exit(1)
            f1, f2, eval, result = batch.contents.get_tensors(device)
            total += batch.contents.size
            def closure() -> Any:
                optimizer.zero_grad()
                output = nnue(f1, f2)
                loss = loss_fn(output, eval, result, exponent, lam) / batch.contents.size
                loss.backward()
                return loss
            nnue.clamp_weights()
            optimizer.step(closure)

        if (save_every > 0 and epoch % save_every == 0) or epoch == epochs:
            save(nnue=nnue, epoch=epoch, lr=lr, gamma=gamma, exponent=exponent, lam=lam, weight_decay=weight_decay, uuid=uuid, filename=filename)


        scheduler.step()
        t = time.time() - t
        eta = time.time() + (epochs - epoch) * t
        print('epoch elapsed %.2f seconds' % (t, ))
        print('estimated time of arrival is %s\n' % (time.strftime('%Y-%m-%d %H:%M', time.localtime(eta)), ))

    print('training elapsed %.2f seconds' % (time.time() - start, ))

def save(nnue: model.NNUE, epoch: int, lr: float, gamma: float, exponent: float, lam: float, weight_decay: float, uuid: customuuid.UUID8, filename: str, dry_run: bool = False) -> None:
    uuid = uuid.set_epoch(epoch)
    filename = filename.replace('uuid', str(uuid))
    filename = filename.replace('epoch', str(epoch))
    filename = filename.replace('lr', str(lr))
    filename = filename.replace('gamma', str(gamma))
    filename = filename.replace('exponent', str(exponent))
    filename = filename.replace('lambda', str(lam))
    filename = filename.replace('weight_decay', str(weight_decay))
    if dry_run:
        print('Will save to the file \'%s\'' % (filename, ))
        return
    print('uuid: %s' % (uuid, ))
    print('epoch: %d' % (epoch, ))
    print('lr: %g' % (lr, ))
    print('gamma: %g' % (gamma, ))
    print('exponent: %g' % (exponent, ))
    print('lambda: %g' % (lam, ))
    print('weight decay: %g' % (weight_decay, ))
    torch.save({'nnue': nnue.state_dict(),
                'epoch': epoch,
                'lr': lr,
                'gamma': gamma,
                'exponent': exponent,
                'lam': lam,
                'weight_decay': weight_decay,
                'uuid': int(uuid)}, filename)

def main() -> None:
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
    parser.add_argument('--filename', type=str, help='Append to UUID, e.g. \'uuid-lambda.ckpt\'\n', default='uuid.ckpt')


    args, unknown = parser.parse_known_args()
    if len(unknown) >= 1:
        if args.training_data is None:
            args.training_data = unknown[0]
        elif args.validation_data is None:
            args.validation_data = unknown[0]
    if len(unknown) >= 2:
        if args.validation_data is None:
            args.validation_data = unknown[1]

    args.filename = args.filename.replace('{lambda}', '{lam}')

    device = torch.device(args.device)
    print('Running on \'%s\'' % (torch.cuda.get_device_name(device), ))

    nnue = model.NNUE().to(device=device, non_blocking=True)

    if args.info and not args.load:
        print('--info without --load')
        sys.exit(1)

    if args.lam < 0.0 or args.lam > 1.0:
        print('lambda must be between 0.0 and 1.0')

    uuid = customuuid.UUID8()

    if args.load:
        if not args.load.endswith('.ckpt'):
            print('Loaded file is not a checkpoint file.')
            sys.exit(2)
        ckpt: dict[str, Any] = torch.load(args.load)
        nnue.load_state_dict(ckpt['nnue'])
        args.start_epoch = ckpt['epoch'] + 1
        # Scale lr according to the scheduler
        args.lr = ckpt['lr'] * (ckpt['gamma'] ** ckpt['epoch'])
        args.gamma = ckpt['gamma']
        args.exponent = ckpt['exponent']
        args.lam = ckpt['lam']
        args.weight_decay = ckpt['weight_decay']
        if 'uuid' in ckpt:
            uuid = customuuid.UUID8(ckpt['uuid'], ckpt['epoch'])
        if args.info:
            if 'uuid' in ckpt:
                print('uuid: %s' % (uuid, ))
            print('epochs: %d' % (ckpt['epoch'], ))
            print('lr: %g (%g)' % (ckpt['lr'], args.lr))
            print('gamma: %g' % (ckpt['gamma'], ))
            print('exponent: %g' % (ckpt['exponent'], ))
            print('lambda: %g' % (ckpt['lam'], ))
            print('weight decay: %g' % (ckpt['weight_decay'], ))
            return

    if batchbit.version() != model.VERSION_NNUE:
        print('version mismatch (%d != %d)' % (batchbit.version(), model.VERSION_NNUE))
        sys.exit(1)

    if args.training_data is None:
        print('need --training-data')
        sys.exit(1)
    if args.validation_data is None:
        print('need --validation-data')
        sys.exit(1)

    save(nnue=nnue, epoch=args.epochs, lr=args.lr, gamma=args.gamma, exponent=args.exponent, lam=args.lam, weight_decay=args.weight_decay, uuid=uuid, filename=args.filename, dry_run=True)

    train_data = batchbit.loader_open(args.training_data.encode(), args.batch_size, args.random_skip, 1, args.lam < 1.0)
    if not train_data:
        sys.exit(1)
    val_data = batchbit.loader_open(args.validation_data.encode(), args.batch_size, 0.0, 1, args.lam < 1.0)
    if not val_data:
        sys.exit(1)

    train(nnue, train_data, val_data, args.start_epoch, args.epochs, args.epoch_size, args.validation_size, device, args.lr, args.gamma, args.exponent, args.save_every, args.lam, args.weight_decay, uuid, args.filename)

    batchbit.loader_close(train_data)
    batchbit.loader_close(val_data)

if __name__ == '__main__':
    main()
