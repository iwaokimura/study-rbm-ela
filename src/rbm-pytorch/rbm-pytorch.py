"""
This is a sample implementation of Restricted Boltzmann machine using PyTorch.
Rererences:
    https://blog.paperspace.com/beginners-guide-to-boltzmann-machines-pytorch/
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

if __name__ == '__main__':
    import torch.optim as optim
    from torch.autograd import Variable
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    def save_show(file_name, img):
        """helper function for save an image."""
        npimg = np.transpose(img.numpy(), (1, 2, 0))
        f = f"./{file_name}.png"
        plt.imshow(npimg)
        plt.imsave(f, npimg)


    rbm = RBM(k = 5)
    train_op = optim.SGD(rbm.parameters(), 0.1)

    batch_size = 64
    num_epoch = 10

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', 
                       train = True,
                       download = True, 
                       transform = transforms.Compose([transforms.ToTensor()])
                       ),
        batch_size = batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
            train = False,
            transform = transforms.Compose([transforms.ToTensor()])),
        batch_size = batch_size
    )
    for epoch in range(num_epoch):
        loss_list = []
        for _, (data, target) in enumerate(train_loader):
            data = Variable(data.view(-1, 784))
            sample_data = data.bernoulli()

            v, v1 = rbm(sample_data)
            loss = rbm.free_energy(v) - rbm.free_energy(v1)
            loss_list.append(loss.data)
            train_op.zero_grad()
            loss.backward()
            train_op.step()
        print(f"Training loss for epoch {epoch}: {np.mean(loss_list)}")

    save_show("real", make_grid(v.view(32, 1, 28, 28).data))
    save_show("generate", make_grid(v1.view(32, 1, 28, 28).data))