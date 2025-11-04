#!/usr/bin/env python3

import torch
import argparse
import struct

from . import model

def quantize(file):
    if not file.endswith('.ckpt'):
        print('Input is not .ckpt')
        return

    nnue = model.nnue()
    nnue.load_state_dict(torch.load(file)['nnue'])
    nnue.clamp_weights()

    out = file.replace('.ckpt', '.nnue')

    with open(out, 'wb') as f:
        f.write(struct.pack('<H', model.VERSION_NNUE))

        tensor = 127 * (2 ** model.FT_SHIFT) * nnue.ft.bias.view(-1)
        mean = torch.mean(torch.abs(tensor)).round().long().item()
        tensor = tensor.round().long()
        print(torch.min(tensor).item(), "<= ft_biases <=", torch.max(tensor).item(), "absolute mean: ", mean)
        bytes = tensor.detach().numpy().astype('<u2').tobytes()
        f.write(bytes)

        weight = 127 * (2 ** model.FT_SHIFT) * nnue.ft.weight.t()
        tensor = weight[:model.FT_IN_DIMS, :]
        virtual = weight[-model.VIRTUAL:, :]
        for i in range(model.FT_IN_DIMS):
            tensor[i] += virtual[i % model.VIRTUAL]
        mean = torch.mean(torch.abs(tensor[:, :model.K_HALF_DIMENSIONS])).round().long().item()
        tensor = tensor.round().long()
        print(torch.min(tensor[:, :256]).item(), "<= ft_weights <=", torch.max(tensor[:, :256]).item(), "absolute mean: ", mean)
        bytes = tensor.detach().numpy().astype('<u2').tobytes()
        f.write(bytes)

        tensor = 127 * (2 ** model.SHIFT) * nnue.hidden1.bias.view(-1)
        mean = torch.mean(torch.abs(tensor)).round().long().item()
        tensor = tensor.round().long()
        print(torch.min(tensor).item(), "<= hidden1_biases <=", torch.max(tensor).item(), "absolute mean: ", mean)
        bytes = tensor.detach().numpy().astype('<u4').tobytes()
        f.write(bytes)

        tensor = (2 ** model.SHIFT) * nnue.hidden1.weight.view(-1)
        mean = torch.mean(torch.abs(tensor)).round().long().item()
        tensor = tensor.round().long()
        print(torch.min(tensor).item(), "<= hidden1_weights <=", torch.max(tensor).item(), "absolute mean: ", mean)
        bytes = tensor.detach().numpy().astype('<u1').tobytes()
        f.write(bytes)

        tensor = 127 * (2 ** model.SHIFT) * nnue.hidden2.bias.view(-1)
        mean = torch.mean(torch.abs(tensor)).round().long().item()
        tensor = tensor.round().long()
        print(torch.min(tensor).item(), "<= hidden2_biases <=", torch.max(tensor).item(), "absolute mean: ", mean)
        bytes = tensor.detach().numpy().astype('<u4').tobytes()
        f.write(bytes)

        tensor = (2 ** model.SHIFT) * nnue.hidden2.weight.view(-1)
        mean = torch.mean(torch.abs(tensor)).round().long().item()
        tensor = tensor.round().long()
        print(torch.min(tensor).item(), "<= hidden2_weights <=", torch.max(tensor).item(), "absolute mean: ", mean)
        bytes = tensor.detach().numpy().astype('<u1').tobytes()
        f.write(bytes)

        tensor = 127 * (2 ** model.SHIFT) * nnue.output.bias.view(-1)
        mean = torch.mean(torch.abs(tensor)).round().long().item()
        tensor = tensor.round().long()
        print(torch.min(tensor).item(), "<= output_biases <=", torch.max(tensor).item(), "absolute mean: ", mean)
        bytes = tensor.detach().numpy().astype('<u4').tobytes()
        f.write(bytes)

        tensor = (2 ** model.SHIFT) * nnue.output.weight.view(-1)
        mean = torch.mean(torch.abs(tensor)).round().long().item()
        tensor = tensor.round().long()
        print(torch.min(tensor).item(), "<= output_weights <=", torch.max(tensor).item(), "absolute mean: ", mean)
        bytes = tensor.detach().numpy().astype('<u1').tobytes()
        f.write(bytes)

        pieces = [ 'Pawn:  ', 'Knight:', 'Bishop:', 'Rook:  ', 'Queen: ' ]

        for piece in range(1, 6):
            print(f'{pieces[piece - 1]}', end='')
            for psqt_bucket in range(0, model.PSQT_BUCKETS):
                average = 0
                for square in range(0, 64):
                    average += weight[model.FT_IN_DIMS + model.make_index_virtual(1, square, piece), model.K_HALF_DIMENSIONS + psqt_bucket].long().item()
                    average -= weight[model.FT_IN_DIMS + model.make_index_virtual(0, square, piece), model.K_HALF_DIMENSIONS + psqt_bucket].long().item()
                average /= 2 * 64
                average = int(average)
                print(f' {average:4d}', end='')
            print('\n', end='')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('file', type=str, nargs='+', help='Checkpoint file(s) (.ckpt)')

    args = parser.parse_args()

    for file in args.file:
        quantize(file)

if __name__ == '__main__':
    main()
