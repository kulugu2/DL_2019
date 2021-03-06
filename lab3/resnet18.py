import dataloader
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0')
learning_rate = 0.0005 #0.0003 
batch_size = 6
epochs = 10      #6000

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = 1
        if in_channel != out_channel:
            self.stride = 2
        self.block = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3 , padding=1, stride = self.stride, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, kernel_size=3 , padding=1, stride = 1, bias=False),
                nn.BatchNorm2d(out_channel)
                )
        self.downsample = None
        if in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channel)
                )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.block(x)
        res = x
        if self.in_channel != self.out_channel:
            res = self.downsample(x)
        out += res
        out = self.relu(out)

        return out

class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        #self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512), BasicBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 ,5)

    def forward(self, x):
        out = self.conv1(x)
        #out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

resnet_model = Resnet18().to(device)
        
def adjust_lr(optimizer, epoch):
    if epoch < 10:
        lr = 0.01
    elif epoch < 500:
        lr = 0.003
    elif epoch < 2000:
        lr = 0.001
    elif epoch < 2500:
        lr = 0.0005
    else:
        lr = 0.0001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

optimizer = torch.optim.SGD(resnet_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
loss_func = nn.CrossEntropyLoss()

def train(model, optimizer):

    model.train()
    train_acc = []
    test_acc = []
    for epoch in range(epochs):
        model.train()
        #adjust_lr(optimizer, epoch)
        correct = 0
        for i, (data, labels) in enumerate(train_loader):
            #data = torch.from_numpy(data)
            data = data.to(device, dtype=torch.float)
            #labels = torch.from_numpy(np.array(labels))
            labels = labels.to(device, dtype=torch.long)

            outputs = model(data)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            _, pred = torch.max(outputs.data, 1)
            correct += (pred == labels).sum().item()
            if i%100 == 0:
                print('finish batch {}'.format(i))
            
        print('Epoch: {} \tLoss: {:.6f}\t Accuracy: {:.2f}'.format(epoch, loss.item(), 100.*correct/len(train_loader.dataset)))
        acc = test(model)
        #if (acc >= 87.0):
        #    torch.save({ 'epoch': epoch, 'state_dict': model.state_dict()}, 'eegnet_model.tar')
        #    break

        train_acc.append(100.* correct/len(train_loader.dataset))
        #acc = test(model)
        test_acc.append(acc)

    return train_acc, test_acc

def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    for i, (data, labels) in enumerate(test_loader):
        #data = torch.from_numpy(data)
        data = data.to(device, dtype=torch.float)
        #labels = torch.from_numpy(np.array(labels))
        labels = labels.to(device, dtype=torch.long)
        with torch.no_grad():
            output = model(data)
        test_loss += loss_func(output, labels).item()
        #print(output.data)
        _, pred = torch.max(output.data, 1)
        #print(pred)
        correct += (pred == labels).sum().item()

    print('Test set: Average loss: {:.4f}, Accuracy:{}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct/len(test_loader.dataset)))

    return 100. * correct/len(test_loader.dataset)


train_dataset = dataloader.RetinopathyLoader('data/', 'train')

test_dataset  = dataloader.RetinopathyLoader('data/', 'test')


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
load_model = Resnet18().to(device)
#import sys
if __name__ == '__main__':
    
    # start training
    #model = Resnet18()
    #print(resnet_model)
    #train_acc, test_acc = train(resnet_model, optimizer)
    #torch.save({ 'state_dict': resnet_model.state_dict()}, 'resnet18_without_pretrain1.tar')
    checkpoint = torch.load('resnet18_with_pretrain_1.tar')
    load_model.load_state_dict(checkpoint['state_dict'])
    test(load_model)
    #for i in range(len(train_acc)):
    #    print(train_acc,test_acc)
    # start testing
    #test(ReLU_model)
