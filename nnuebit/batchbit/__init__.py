#!/usr/bin/env python3

import ctypes

from .batchbit import batch

__all__ = ['batch']

lib = ctypes.cdll.LoadLibrary('libbatchbit.so')

lib.batchbit_init.argtypes = None
lib.batchbit_init.restype = None
lib.batchbit_init()

batch_fetch = lib.batch_fetch
batch_fetch.argtypes = [ctypes.c_void_p]
lib.batch_fetch.restype = ctypes.POINTER(batchbit.batch)

loader_open = lib.loader_open
loader_open.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_double]
loader_open.restype = ctypes.c_void_p

loader_reset = lib.loader_reset
loader_reset.argtypes = [ctypes.c_void_p]
loader_reset.restype = None

loader_close = lib.loader_close
loader_close.argtypes = [ctypes.c_void_p]
loader_close.restype = None
