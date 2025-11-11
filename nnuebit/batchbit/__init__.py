#!/usr/bin/env python3

import ctypes

from .batchbit import Batch

__all__ = ['batchbit']

lib = ctypes.cdll.LoadLibrary('libbatchbit.so')

lib.batchbit_init.argtypes = []
lib.batchbit_init.restype = None
lib.batchbit_init()

batch_fetch = lib.batch_fetch
batch_fetch.argtypes = [ctypes.c_void_p]
batch_fetch.restype = ctypes.POINTER(Batch)

batch_free = lib.batch_free
batch_free.argtypes = [ctypes.POINTER(Batch)]
batch_free.restype = None

loader_open = lib.loader_open
loader_open.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int]
loader_open.restype = ctypes.c_void_p

loader_close = lib.loader_close
loader_close.argtypes = [ctypes.c_void_p]
loader_close.restype = None

version = lib.batchbit_version
version.argtypes = []
version.restype = ctypes.c_int
