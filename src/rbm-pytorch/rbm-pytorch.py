from rbm.rbm import RBM

if __name__ == '__main__':
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
    train_op = optim.SGD(rbm.parameters(), 0.1)

    batch_size = 64
    num_epoch = 10

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', 
            train = True,
            download = True, 
            transform = transforms.Compose([transforms.ToTensor()])),
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