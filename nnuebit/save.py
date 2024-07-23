#!/usr/bin/env python3

import torch
import datetime

def save(nnue):
    name = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ.pt')

    torch.save(nnue.state_dict(), name)

    return name
