#!/usr/bin/env python3

import math
import torch
import time

from . import model
from . import batchbit

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

def run(nnue, train_data, val_data, epochs, device, lr, gamma, exponent, save_every):
    start = time.time()

    optimizer = torch.optim.Adam(nnue.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epochs + 1):
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
            torch.save(nnue.state_dict(), 'temp.pt')

        scheduler.step()
        t = time.time() - t
        eta = time.time() + (epochs - epoch) * t
        print(f'epoch elapsed {round(t, 2)} seconds')
        print(f'estimated time of arrival is {time.strftime('%Y-%m-%d %H:%M', time.localtime(eta))}\n')

    print(f'training elapsed {round(time.time() - start, 2)} seconds')
