import dataloader
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda:0')
learning_rate = 0.005 
batch_size = 1080
epochs = 100

class dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.data)

class deepconvnet(nn.Module):
    def __init__(self, activation='ELU'):
        super().__init__()
        self.firstconv = nn.Sequential(
                nn.Conv2d(1, 25, kernel_size=(1, 5)),
                nn.Conv2d(25, 25, kernel_size=(2, 1)),
                nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, track_running_stats=True),
                nn.ELU() if activation=='ELU' else nn.ReLU() if activation=='ReLU' else nn.LeakyReLU(),
                #nn.ELU(alpha=1.0),
                #nn.ReLU(),
                #nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5)
                )
        self.secconv = nn.Sequential(
                nn.Conv2d(25, 50, kernel_size=(1, 5)),
                nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, track_running_stats=True),
                nn.ELU() if activation=='ELU' else nn.ReLU() if activation=='ReLU' else nn.LeakyReLU(),
                #nn.ELU(alpha=1.0),
                #nn.ReLU(),
                #nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5)
                )
        self.thridconv = nn.Sequential(
                nn.Conv2d(50, 100, kernel_size=(1, 5)),
                nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, track_running_stats=True),
                nn.ELU() if activation=='ELU' else nn.ReLU() if activation=='ReLU' else nn.LeakyReLU(),
                #nn.ELU(alpha=1.0),
                #nn.ReLU(),
                #nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5)
                )
        self.fourthconv = nn.Sequential(
                nn.Conv2d(100, 200, kernel_size=(1, 5)),
                nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, track_running_stats=True),
                nn.ELU() if activation=='ELU' else nn.ReLU() if activation=='ReLU' else nn.LeakyReLU(),
                #nn.ELU(alpha=1.0),
                #nn.ReLU(),
                #nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5)
                )
        self.classify = nn.Sequential(
                nn.Linear(in_features=8600, out_features=2, bias=True)
                )

    def forward(self, x):
        #print(x.shape)
        out = self.firstconv(x)
        #print(out.shape)
        out = self.secconv(out)
        #print(out.shape)
        out = self.thridconv(out)
        #print(out.shape)
        out = self.fourthconv(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        out = self.classify(out)
        return out
def adjust_lr(optimizer, epoch):
    if epoch < 10:
        lr = 0.01
    elif epoch < 500:
        lr = 0.001
    elif epoch < 1000:
        lr = 0.0005
    else:
        lr = 0.00005

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

ReLU_model = deepconvnet(activation='ReLU').to(device)
ReLU_optimizer = torch.optim.Adam(ReLU_model.parameters(), lr=learning_rate)
ELU_model = deepconvnet(activation='ELU').to(device)
LReLU_model = deepconvnet(activation='Leaky_ReLU').to(device)
ELU_optimizer = torch.optim.Adam(ELU_model.parameters(), lr=learning_rate)
LReLU_optimizer = torch.optim.Adam(LReLU_model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()


def test(model):

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

    print('Test set: Average loss: {:.4f}, Accuracy:{}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct/len(test_loader.dataset)))

    return 100. * correct/len(test_loader.dataset)

def train(model, optimizer):
    # start training
    model.train()
    train_acc = []
    test_acc = []
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
        train_acc.append(100.* correct/len(train_loader.dataset))
        acc = test(model)
        test_acc.append(acc)

    return train_acc, test_acc

train_data, train_label, test_data, test_label = dataloader.read_bci_data() 

train_dataset = dataset(train_data, train_label)
test_dataset  = dataset(test_data, test_label)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if __name__ == '__main__':
    
    
    x_axis = np.arange(epochs)
    # start training
    LReLU_train_acc, LReLU_test_acc = train(LReLU_model, LReLU_optimizer)    
    ReLU_train_acc, ReLU_test_acc = train(ReLU_model, ReLU_optimizer)    
    ELU_train_acc, ELU_test_acc = train(ELU_model, ELU_optimizer)    
    # start testing
    test(LReLU_model)

    plt.plot(x_axis, LReLU_train_acc, 'r', label='leaky_relu_train')
    plt.plot(x_axis, LReLU_test_acc, 'b', label='leaky_relu_test')

    plt.plot(x_axis, ReLU_train_acc, 'g', label='relu_train')
    plt.plot(x_axis, ReLU_test_acc, 'c', label='relu_test')
    plt.plot(x_axis, ELU_train_acc, 'y', label='ELU_train')
    plt.plot(x_axis, ELU_test_acc, 'k', label='ELU_test')
    plt.legend()
    plt.show()
    
    '''
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

    print('\nTest set: Average loss: {:.4f}, Accuracy:{}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct/len(test_loader.dataset)))
    '''
 
