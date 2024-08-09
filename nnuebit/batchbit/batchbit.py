#!/usr/bin/env python3

import numpy
import ctypes
import torch

from .. import model

class batch(ctypes.Structure):
    _fields_ = [
            ('actual_size', ctypes.c_size_t),
            ('ind_active', ctypes.c_int),
            ('ind1', ctypes.POINTER(ctypes.c_int32)),
            ('ind2', ctypes.POINTER(ctypes.c_int32)),
            ('eval', ctypes.POINTER(ctypes.c_float)),
    ]

    def get_tensors(self, device):
        eval = torch.from_numpy(numpy.ctypeslib.as_array(self.eval, shape=(self.actual_size, 1))).pin_memory().to(device=device, non_blocking=True)

        val1 = torch.ones(self.ind_active).pin_memory().to(device=device, non_blocking=True)
        val2 = torch.ones(self.ind_active).pin_memory().to(device=device, non_blocking=True)

        ind1 = torch.transpose(torch.from_numpy(numpy.ctypeslib.as_array(self.ind1, shape=(self.ind_active, 2))), 0, 1).pin_memory().to(device=device, non_blocking=True)
        ind2 = torch.transpose(torch.from_numpy(numpy.ctypeslib.as_array(self.ind2, shape=(self.ind_active, 2))), 0, 1).pin_memory().to(device=device, non_blocking=True)

        f1 = torch.sparse_coo_tensor(ind1, val1, (self.actual_size, model.FT_IN_DIMS + model.VIRTUAL), check_invariants=False).to(device=device, non_blocking=True)
        f2 = torch.sparse_coo_tensor(ind2, val2, (self.actual_size, model.FT_IN_DIMS + model.VIRTUAL), check_invariants=False).to(device=device, non_blocking=True)

        f1._coalesced_(True)
        f2._coalesced_(True)

        return f1, f2, eval

    def is_empty(self):
        return (self.actual_size == 0)
