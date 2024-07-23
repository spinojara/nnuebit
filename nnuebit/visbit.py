#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import argparse
import ctypes

def visualize_ft(name, lib):
    image = np.empty((2560, 4096), dtype=np.int32)
    lib.image_ft(image)
    mean = image.mean()
    tmax = np.percentile(image, 95)
    tmin = 0

    plt.imshow(image, aspect='auto', cmap='viridis', vmin=tmin, vmax=tmax, interpolation='bilinear')
    for i in range(1, 8):
        plt.plot([0, 4095], [i * 40 * 8, i * 40 * 8], color='red')
    for i in range(1, 32):
        plt.plot([i * 16 * 8, i * 16 * 8], [0, 2559], color='red')
    plt.colorbar()
    plt.axis('off')
    plt.title(f'Input Weights {name}')

def visualize_psqt(name, lib):
    fig, axs = plt.subplots(1, 5)
    pieces = [ 'Pawn', 'Knight', 'Bishop', 'Rook', 'Queen' ]
    image = np.empty((8, 8), dtype=np.int32)
    for i, ax in enumerate(axs.flatten()):
        lib.image_psqt(image, 1 + i)
        im = ax.imshow(image, aspect='auto', cmap='viridis')
        plt.colorbar(im, orientation='horizontal', pad=0.05)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{pieces[i % 5]}')
    plt.suptitle(f'Psqt Weights {name}')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('file', type=str, help='Network file (.nnue)')
    parser.add_argument('--psqt', action='store_true', help='Visualize psqt instead')

    args = parser.parse_args()

    lib = ctypes.cdll.LoadLibrary('libvisbit.so')
    
    lib.read_ft_weights.argtypes = [ctypes.c_char_p]
    lib.read_ft_weights.restype = None
    
    lib.image_ft.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")]
    lib.image_ft.restype = None
    
    lib.image_psqt.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"), ctypes.c_int]
    lib.image_psqt.restype = None

    lib.read_ft_weights(ctypes.c_char_p(args.file.encode()))
    
    if not args.psqt:
        visualize_ft(args.file, lib)
    else:
        visualize_psqt(args.file, lib)

    plt.show()

if __name__ == '__main__':
    main()
#def visualize_ft(name):
#    image = np.empty((2560, 4096), dtype = np.int32)
#    lib.image_ft(image)
#    mean = image.mean()
#    tmax = np.percentile(image, 95)
#    tmin = 0
#
#    plt.imshow(image, aspect = 'auto', cmap = 'viridis', vmin = tmin, vmax = tmax, interpolation = 'bilinear')
#    for i in range(1, 8):
#        plt.plot([0, 4095], [i * 40 * 8, i * 40 * 8], color = 'red')
#    for i in range(1, 32):
#        plt.plot([i * 16 * 8, i * 16 * 8], [0, 2559], color = 'red')
#    plt.colorbar()
#    plt.axis('off')
#    plt.title(f'Input Weights {name}')
#
#import sys
#def visualize_psqt(name):
#    fig, axs = plt.subplots(1, 5)
#    pieces = [ 'Pawn', 'Knight', 'Bishop', 'Rook', 'Queen' ]
#    image = np.empty((8, 8), dtype = np.int32)
#    for i, ax in enumerate(axs.flatten()):
#        lib.image_psqt(image, 1 + i)
#        im = ax.imshow(image, aspect = 'auto', cmap = 'viridis')
#        plt.colorbar(im, orientation = 'horizontal', pad = 0.05)
#        ax.set_aspect('equal')
#        ax.axis('off')
#        ax.set_title(f'{pieces[i % 5]}')
#    plt.suptitle(f'Psqt Weights {name}')
#
#if len(sys.argv) > 1:
#    name = sys.argv[1]
#    lib.read_ft_weights(ctypes.c_char_p(name.encode()))
#    visualize_ft(name)
#    plt.show()
#else:
#    print("No filename provided")
