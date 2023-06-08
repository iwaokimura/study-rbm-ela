from rbm.rbm import RBM

import numpy as np
import torch
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
with open("./rbm_model.dat", "rb") as f:
    rbm.load_state_dict(torch.load(f))
    rbm.eval()
v1 = torch.tensor(np.random.rand(728))
v1.bernoulli()
_, v1 = rbm(v1)
save_show("generate", make_grid(v1.view(32, 1, 28, 28).data))
