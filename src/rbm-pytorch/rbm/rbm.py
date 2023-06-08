"""
This is a sample implementation of Restricted Boltzmann machine using PyTorch.
Rererences:
    https://blog.paperspace.com/beginners-guide-to-boltzmann-machines-pytorch/
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RBM(torch.nn.Module):
    def __init__(self, n_visible = 784, n_hidden = 500, k = 5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.k = k

    def sample_from_p(self, p):
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))
    
    def v_to_h(self, v):
        p_h = F.sigmoid(F.linear(v, self.W.t(), self.h_bias)) # XXX fixme!
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h
    
    def h_to_v(self, h):
        p_v = F.sigmoid(F.linear(h, self.W, self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v
    
    def forward(self, v):
        pre_h1, h1 = self.v_to_h(v)
        
        v_ = 0.0
        h_ = h1
        for _ in range(self.k):
            prev_v_, v_ = self.h_to_v(h_)
            pre_h_, h_ = self.v_to_h(v_)

        return v, v_
    
    def free_energy(self, v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W.t(), self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        rv = (-hidden_term - vbias_term).mean()
        # print(f"debug@free_energy(): rv = {rv}")
        return(rv)

    def save(self, path_f):
        torch.save(self.state_dict(), path_f)
