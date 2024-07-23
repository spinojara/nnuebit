#!/usr/bin/env python3

import torch
import ctypes
import time
import math
import argparse

from . import batchbit
from . import model
from . import train
from . import quantize
from . import save

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('training_data', type=str, help='Training data file')
    parser.add_argument('validation_data', type=str, help='Validation data file')
    parser.add_argument('--random-skip', type=float, help='Random skipping frequency', default=0.8)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=8192)
    parser.add_argument('--device', type=str, help='Pytorch device', default='cuda')
    parser.add_argument('--load', type=str, help='Load old network file (.pt)', default='')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=400)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)

    args = parser.parse_args()

    nnue = model.nnue().to(device=args.device, non_blocking=True)

    if args.load != '':
        nnue.load_state_dict(torch.load(args.load))

    train_data = batchbit.batch_open(args.training_data.encode(), args.batch_size, args.random_skip)
    val_data = batchbit.batch_open(args.validation_data.encode(), args.batch_size, 0.0)

    train.run(nnue, train_data, val_data, args.epochs, args.device, args.lr)

    name = save.save(nnue)
    quantize.quantize(name)

    batchbit.batch_close(train_data)
    batchbit.batch_close(val_data)

if __name__ == '__main__':
    main()