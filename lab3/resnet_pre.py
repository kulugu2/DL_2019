import dataloader
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import models
import plot_confusion_matrix as cm

device = torch.device('cuda:0')
learning_rate = 1e-3 #0.0003 
batch_size = 8
epochs = 20     #6000

outfile = open('tt.txt','w')

resnet_model = models.ResNet().to(device)
        

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
                outfile.write('finish batch {}\n'.format(i))
                #print(loss)
                #print(pred, labels)
            
        print('Epoch: {} \tLoss: {:.6f}\t Accuracy: {:.2f}'.format(epoch, loss.item(), 100.*correct/len(train_loader.dataset)))
        outfile.write('Epoch: {} \tLoss: {:.6f}\t Accuracy: {:.2f}\n'.format(epoch, loss.item(), 100.*correct/len(train_loader.dataset)))
        acc = test(model)
        if (acc >= 82.0):
            torch.save({ 'epoch': epoch, 'state_dict': model.state_dict()}, 'resnet18_best_model.tar')
            break

        train_acc.append(100.* correct/len(train_loader.dataset))
        #acc = test(model)
        test_acc.append(acc)

    return train_acc, test_acc

def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    preds = []
    for i, (data, labels) in enumerate(test_loader):
        #data = torch.from_numpy(data)
        data = data.to(device, dtype=torch.float)
        #labels = torch.from_numpy(np.array(labels))
        labels = labels.to(device, dtype=torch.long)
        with torch.no_grad():
            output = model(data)
        test_loss += loss_func(output, labels).item()
        _, pred = torch.max(output.data, 1)
        correct += (pred == labels).sum().item()
        preds.extend(pred.tolist())

    print('Test set: Average loss: {:.4f}, Accuracy:{}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct/len(test_loader.dataset)))

    outfile.write('Test set: Average loss: {:.4f}, Accuracy:{}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct/len(test_loader.dataset)))
    cm.plot_confusion_matrix(np.array(preds), np.array(test_dataset.label), np.array(['0','1','2','3','4']), normalize=True)
    return 100. * correct/len(test_loader.dataset)


train_dataset = dataloader.RetinopathyLoader('data/', 'train')

test_dataset  = dataloader.RetinopathyLoader('data/', 'test')


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

load_model = models.ResNet().to(device)
#import sys
if __name__ == '__main__':
    
    # start training
    #model = Resnet18()
    #print(resnet_model)
    #train_acc, test_acc = train(resnet_model, optimizer)
    #torch.save({ 'state_dict': resnet_model.state_dict()}, 'resnet18_with_pretrain_1.tar')
   
    checkpoint = torch.load('resnet50_best.tar')
    load_model.load_state_dict(checkpoint['state_dict'])
    test(load_model)
    #print(train_acc,test_acc)
    #print(train_acc, file=outfile)
    #outfile.write('\n')
    #print(test_acc, file=outfile)
    #outfile.write('\n')
    #outfile.close()
    # start testing
    #test(ReLU_model)
