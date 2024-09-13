#!/usr/bin/env python3

import torch
import random
import sys

VERSION_NNUE = 2

K_HALF_DIMENSIONS = 512
FT_OUT_DIMS = 2 * K_HALF_DIMENSIONS
FV_SCALE = 16

SHIFT = 6
FT_SHIFT = 0

weight_limit = 127 / 2 ** SHIFT

PS_W_PAWN   =  0 * 64
PS_B_PAWN   =  1 * 64
PS_W_KNIGHT =  2 * 64
PS_B_KNIGHT =  3 * 64
PS_W_BISHOP =  4 * 64
PS_B_BISHOP =  5 * 64
PS_W_ROOK   =  6 * 64
PS_B_ROOK   =  7 * 64
PS_W_QUEEN  =  8 * 64
PS_B_QUEEN  =  9 * 64
PS_KING     = 10 * 64
PS_END      = 11 * 64

FT_IN_DIMS = 32 * PS_END
VIRTUAL = PS_END

piece_to_index = [
	[ 0, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING,
	     PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, ],
	[ 0, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING,
	     PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, ],
]

piece_value = [ 0, 97, 491, 514, 609, 1374, ]

def orient(turn, square):
    return square ^ (0x0 if turn else 0x38) 

def make_index_virtual(turn, square, piece):
    return orient(turn, square) + piece_to_index[turn][piece]

class nnue(torch.nn.Module):
    def __init__(self):
        super(nnue, self).__init__()
        self.buckets = 8
        self.ft = torch.nn.Linear(FT_IN_DIMS + VIRTUAL, K_HALF_DIMENSIONS + self.buckets)
        self.hidden1 = torch.nn.Linear(FT_OUT_DIMS, 16 * self.buckets)
        self.hidden2 = torch.nn.Linear(16, 32 * self.buckets)
        self.output = torch.nn.Linear(32, self.buckets)
        self.offset = None

        # Initialize virtual features to 0
        torch.nn.init.zeros_(self.ft.weight[:, -VIRTUAL:])

        # Psqt Values
        for bucket in range(0, self.buckets):
            for color in range(0, 2):
                for piece in range(1, 6):
                    for square in range(64):
                        self.ft.weight.data[K_HALF_DIMENSIONS + bucket, FT_IN_DIMS + make_index_virtual(color, square, piece)] = (2 * color - 1) * piece_value[piece] / (127 * 2 ** FT_SHIFT)

        # Initialize output bias to 0
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, bucket, features1, features2):
        if self.offset == None or self.offset.shape[0] != bucket.shape[0]:
            self.offset = torch.arange(0, bucket.shape[0] * self.buckets, self.buckets, device=bucket.device)

        idx = bucket.flatten() + self.offset

        f1, psqt1 = torch.split(self.ft(features1), [K_HALF_DIMENSIONS, self.buckets], dim=1)
        f2, psqt2 = torch.split(self.ft(features2), [K_HALF_DIMENSIONS, self.buckets], dim=1)

        psqtaccumulation = 0.5 * (psqt1.gather(1, bucket) - psqt2.gather(1, bucket))
        accumulation = torch.cat([f1, f2], dim=1)
        ft_out = self.clamp(accumulation)

        hidden1_out = self.hidden1(ft_out).reshape(-1, self.buckets, 16)
        hidden1_out = self.clamp(hidden1_out.view(-1, 16)[idx])

        hidden2_out = self.hidden2(hidden1_out).reshape(-1, self.buckets, 32)
        hidden2_out = self.clamp(hidden2_out.view(-1, 32)[idx])

        return self.output(hidden2_out).gather(1, bucket) + FV_SCALE * 2 ** FT_SHIFT * psqtaccumulation / 2 ** SHIFT

    def clamp(self, x):
        return x.clamp_(0.0, 1.0)

    def clamp_weights(self):
        self.hidden1.weight.data.clamp_(-weight_limit, weight_limit)
        self.hidden2.weight.data.clamp_(-weight_limit, weight_limit)
        self.output.weight.data.clamp_(-weight_limit, weight_limit)
