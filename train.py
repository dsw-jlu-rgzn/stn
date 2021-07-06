from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import stn_model
import argparse
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/mnist_number')
plt.ion()
#use_cuda = torch.cuda.is_available()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=0)

#model = stn_model.Net()
#if use_cuda:
#    model.cuda()
#optimizer = optim.SGD(model.parameters(),lr=0.01)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100 * batch_idx / len(train_loader), loss.data))


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        #累加loss
        test_loss += F.nll_loss(output, target, size_average=False).data
        pred = output.data.max(1, keepdim=True)[1]
        correct +=pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /=len(test_loader.dataset)
    writer.add_scalar('texting loss', test_loss, epoch * len(test_loader))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))
def convert_image_np(inp):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp
def visualize_stn():
    data, _ = next(iter(test_loader))
    data = Variable(data, volatile=True)

    if use_cuda:
        data = data.cuda()
    input_tensor = data.cpu().data
    transformed_input_tensor = model.stn(data).cpu().data
    in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))
    out_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor))
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(in_grid)
    axarr[0].set_title('Dataset Images')

    axarr[1].imshow(out_grid)
    axarr[1].set_title('Transformed Images')
def print_args(parser, args, only_non_defaut=False):
    default_str_list = ['====default args====']
    non_default_str_list = ['====not default args====']
    args_dict = vars(args)
    for k, v in args_dict.items():
        default = parser.get_default(k)
        if v == default:
            default_str_list.append('{}:{}'.format(k, v))
        else:
            non_default_str_list.append('{}:{} (default:{})'.format(k, v, default))
    default_str = '\n'.join(default_str_list)
    non_default_str_list = '\n'.join(non_default_str_list)
    print(non_default_str_list)
    if not only_non_defaut:
        print(default_str)
        print('-'*15)
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',default=0.01, type=float, help='learning rate(default:%(default)s)')
    parser.add_argument('--epoch',default=20, type=int)
    args = parser.parse_args()
    print_args(parser, args, only_non_defaut=False)
    use_cuda = torch.cuda.is_available()
    model = stn_model.Net()
    print('the cuda is:')
    print(use_cuda)
    print('\n'+'-'*15)
    if use_cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epoch+1):
        train(epoch)
        test(epoch)
    visualize_stn()
    plt.ioff()
    plt.show()