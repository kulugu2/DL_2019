import dataloader
import torch
import torch.nn as nn
import torch.utils.data


device = torch.device('cuda:0')
learning_rate = 0.0003 #0.0003 
batch_size = 1080
epochs = 6000      #6000

class dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.data)

class EEGnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.firstconv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
                nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
        self.depthwiseconv = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ELU(alpha=1.0),
                nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
                nn.Dropout(p=0.2)
                )
        self.separableConv = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ELU(alpha=1.0),
                nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
                nn.Dropout(p=0.7)
                )
        self.classify = nn.Sequential(
                nn.Linear(in_features=736, out_features=2, bias=True)
                )

    def forward(self, x):
        out = self.firstconv(x)
        out = self.depthwiseconv(out)
        out = self.separableConv(out)
        out = out.reshape(out.size(0), -1)
        out = self.classify(out)
        return out
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

def test():
    model.eval()
    test_loss = 0
    correct = 0

    for i, (data, labels) in enumerate(test_loader):
        data = data.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        with torch.no_grad():
            output = model(data)
        test_loss += loss_func(output, labels).item()
        _, pred = torch.max(output.data, 1)
        correct += (pred == labels).sum().item()

    print('Test set: Average loss: {:.4f}, Accuracy:{}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct/len(test_loader.dataset)))

    return 100. * correct/len(test_loader.dataset)

train_data, train_label, test_data, test_label = dataloader.read_bci_data() 

train_dataset = dataset(train_data, train_label)
test_dataset  = dataset(test_data, test_label)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = EEGnet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()
if __name__ == '__main__':
    
    #train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    
    #print(len(train_data))
    #model = EEGnet().to(device)
    #print(model)
    
    # start training
    model.train()
    for epoch in range(epochs):
        model.train()
        #adjust_lr(optimizer, epoch)
        correct = 0
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(data)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            _, pred = torch.max(outputs.data, 1)
            correct += (pred == labels).sum().item()
            
        print('Epoch: {} \tLoss: {:.6f}\t Accuracy: {:.2f}'.format(epoch, loss.item(), 100.*correct/len(train_loader.dataset)))
        acc = test()
        if (acc >= 86.0):
            torch.save({ 'epoch': epoch, 'state_dict': model.state_dict()}, 'eegnet_model.tar')
            break

    # start testing
    model.eval()
    test_loss = 0
    correct = 0

    for i, (data, labels) in enumerate(test_loader):
        data = data.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        with torch.no_grad():
            output = model(data)
        test_loss += loss_func(output, labels).item()
        _, pred = torch.max(output.data, 1)
        correct += (pred == labels).sum().item()

    print('\nTest set: Average loss: {:.4f}, Accuracy:{}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct/len(test_loader.dataset)))

 
