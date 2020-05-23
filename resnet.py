import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual block


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=900):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(5, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(1024, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return torch.sigmoid(out)


def IoU(outputs, targets, smooth=1e-6):
    intersection = ((outputs > 0) & (targets > 0)).float().sum(1)
    union = ((outputs > 0) | (targets > 0)).float().sum(1)
    iou = (intersection + smooth) / (union + smooth)
    return iou


class WildFireDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]

    def __len__(self):
        return self.x.shape[0]


resmodel1 = ResNet(ResidualBlock, [2, 2, 2]).to(device)
resmodel2 = ResNet(ResidualBlock, [2, 2, 2]).to(device)

criterion = nn.MSELoss()
optimizer1 = torch.optim.Adam(resmodel1.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(resmodel2.parameters(), lr=0.001)

DATASET_PATH = 'data/uci_ml_hackathon_fire_dataset_2012-05-09_2013-01-01_30k_train_v2.hdf5'

with h5py.File(DATASET_PATH, 'r') as f:
    train_data = {}
    for k in list(f):
        train_data[k] = f[k][:]

print('dataset loaded')
x = train_data['observed']
y = train_data['target']

ytemp = np.zeros((30000, 1800))
ytemp[:, :900] = y[:, 0, ...].reshape((30000, 900))
ytemp[:, 900:] = y[:, 1, ...].reshape((30000, 900))
y = ytemp


split = np.arange(30000)
np.random.shuffle(split)

train_ind = split[:24000]
valid_ind = split[24000:]

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_ind)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_ind)

dataset1 = WildFireDataset(x, y[:, :900])
dataset2 = WildFireDataset(x, y[:, 900:])

train_loader1 = torch.utils.data.DataLoader(
    dataset1, batch_size=32, sampler=train_sampler)
valid_loader1 = torch.utils.data.DataLoader(
    dataset1, batch_size=32, sampler=valid_sampler)
train_loader2 = torch.utils.data.DataLoader(
    dataset2, batch_size=32, sampler=train_sampler)
valid_loader2 = torch.utils.data.DataLoader(
    dataset2, batch_size=32, sampler=valid_sampler)
print('train test split finished')


def train(resmodel, optimizer, criterion, train_loader, valid_loader, name='0'):
    writer = SummaryWriter(log_dir='log/resnet_{}/'.format(name))

    print('start training')
    best_iou = 0.0
    best_valid_loss = 10000

    for epoch in range(50):
        total_loss = 0
        total_iou = 0
        resmodel.train(True)
        for batch, (prev, target) in enumerate(train_loader):
            inp = torch.autograd.Variable(
                torch.Tensor(prev.float())).to(device)
            target = torch.autograd.Variable(
                torch.Tensor(target.float())).to(device)
            optimizer.zero_grad()
            out = resmodel(inp)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_iou += IoU(out, target).sum()

        epoch_train_iou = total_iou / 24000
        epoch_train_loss = total_loss / 24000
        print('iter {} finish, training loss is {}'.format(
            epoch, epoch_train_loss))
        print('iter {} finish, training iou is {}'.format(epoch, epoch_train_iou))
        writer.add_scalar('Train/Loss', epoch_train_loss, epoch)
        writer.add_scalar('Train/Mean IoU', epoch_train_iou, epoch)

        with torch.no_grad():
            total_loss = 0
            total_iou = 0
            resmodel.eval()
            for batch, (prev, target) in enumerate(valid_loader):
                inp = torch.autograd.Variable(
                    torch.Tensor(prev.float())).to(device)
                target = torch.autograd.Variable(
                    torch.Tensor(target.float())).to(device)
                optimizer.zero_grad()
                out = resmodel(inp)
                loss = criterion(out, target)
                total_loss += loss.item()
                total_iou += IoU(out, target).sum()

            epoch_valid_iou = total_iou / 6000
            epoch_valid_loss = total_loss / 6000
            print('iter {} finish, validation loss is {}'.format(
                epoch, epoch_valid_loss))
            print('iter {} finish, validation iou is {}'.format(
                epoch, epoch_valid_iou))
            writer.add_scalar('Valid/Loss', epoch_valid_loss, epoch)
            writer.add_scalar('Valid/Mean IoU', epoch_valid_iou, epoch)

        torch.save({
            'epoch': epoch,
            'model_state_dict': resmodel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            'model/resnet_{}/{}th'.format(name, epoch))

        if epoch_valid_iou > best_iou:
            best_iou = epoch_valid_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': resmodel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                'model/resnet_{}/best_iou'.format(name))

        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': resmodel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                'model/resnet_{}/best_valid_loss'.format(name))


if __name__ == "__main__":
    train(resmodel2, optimizer2, criterion,
          train_loader2, valid_loader2, name='12')
