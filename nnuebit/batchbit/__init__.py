#!/usr/bin/env python3

import ctypes

from .batchbit import batch

__all__ = ['batch']

lib = ctypes.cdll.LoadLibrary('libbatchbit.so')

lib.batch_init.argtypes = None
lib.batch_init.restype = None
lib.batch_init()

next_batch = lib.next_batch
next_batch.argtypes = [ctypes.c_void_p]
lib.next_batch.restype = ctypes.POINTER(batchbit.batch)

batch_open = lib.batch_open
batch_open.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_double]
batch_open.restype = ctypes.c_void_p

batch_reset = lib.batch_reset
batch_reset.argtypes = [ctypes.c_void_p]
batch_reset.restype = None

batch_close = lib.batch_close
batch_close.argtypes = [ctypes.c_void_p]
batch_close.restype = None
