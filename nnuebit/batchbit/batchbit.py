#!/usr/bin/env python3

import numpy
import ctypes
import torch
from typing import Tuple

from .. import model

class Batch(ctypes.Structure):
    _fields_ = [
            ('size', ctypes.c_size_t),
            ('ind_active', ctypes.c_int),
            ('ind1', ctypes.POINTER(ctypes.c_int32)),
            ('ind2', ctypes.POINTER(ctypes.c_int32)),
            ('eval', ctypes.POINTER(ctypes.c_float)),
            ('result', ctypes.POINTER(ctypes.c_float)),
            ('_next', ctypes.c_void_p),
    ]

    def get_tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eval = torch.tensor(numpy.ctypeslib.as_array(self.eval, shape=(self.size, 1)), device=device)
        result = torch.tensor(numpy.ctypeslib.as_array(self.result, shape=(self.size, 1)), device=device)

        val = torch.ones(self.ind_active, device=device)

        ind1 = torch.transpose(torch.tensor(numpy.ctypeslib.as_array(self.ind1, shape=(self.ind_active, 2)), device=device), 0, 1)
        ind2 = torch.transpose(torch.tensor(numpy.ctypeslib.as_array(self.ind2, shape=(self.ind_active, 2)), device=device), 0, 1)

        f1 = torch.sparse_coo_tensor(ind1, val, (self.size, model.FT_IN_DIMS + model.VIRTUAL), check_invariants=False, is_coalesced=True)
        f2 = torch.sparse_coo_tensor(ind2, val, (self.size, model.FT_IN_DIMS + model.VIRTUAL), check_invariants=False, is_coalesced=True)

        return f1, f2, eval, result
