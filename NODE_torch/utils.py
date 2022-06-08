from turtle import down
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import Callback

def get_MNIST_dloaders(batch_size=64, size=28, path='./data/mnist_data',download = True, num_workers=8):
    
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    train_data = datasets.MNIST(path, train=True,
                                download=download, transform=all_transforms)
    test_data = datasets.MNIST(path, train=False,
                                download=download, transform=all_transforms)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers= num_workers)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers= num_workers)
    
    return trainloader, testloader

def get_cifar_dloaders(batch_size=64, size=32, path='./data/cifar10_data',download = True, num_workers=8):

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10(path, train=True, download=download, 
                                    transform=transform_train)
    test_data = datasets.CIFAR10(path, train=False, download=download,
                                    transform=transform_test)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers= num_workers)
    
    return trainloader, testloader

class MNISTLearner(pl.LightningModule):
    def __init__(self, model:nn.Module, trainloader, testloader):
        super().__init__()
        self.lr = 1e-3
        self.model = model
        self.iters = 0.
        self.trainloader, self.testloader = trainloader, testloader
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        self.iters += 1.
        x, y = batch   
        # x, y = x.to(device), y.to(device)
        # y_hat = self.model(x)\
        self.model[1].vf.nfe = 0
        _, trajectory = self.model[:2](x) # In ode net, the output is (shape_of_t_span, true_output )
        y_hat = self.model[2:](trajectory[-1]) 
        loss = nn.CrossEntropyLoss()(y_hat, y)
        # epoch_progress = self.iters / self.loader_len
        acc = Accuracy()(y_hat, y)
        nfe = self.model[1].vf.nfe ; self.model[1].vf.nfe = 0
        tqdm_dict = {
            'train_loss': loss, 
            'accuracy': acc, 
            'NFE': torch.tensor(nfe, dtype=torch.float) }
        # logs = {'train_loss': loss, 'epoch': epoch_progress}
        self.log_dict({
            'epoch': self.current_epoch, 
            'train_loss': loss, 
            'accuracy': acc, 
            'NFE': torch.tensor(nfe, dtype=torch.float)
            }, on_epoch=True, on_step=False)
        # self.log_dict(tqdm_dict, on_epoch=True, on_step=False)
        return {
            'loss': loss, 
            'progress_bar': tqdm_dict, 
            # 'log': logs
        }   

    def test_step(self, batch, batch_nb):
        x, y = batch
        # x, y = x.to(device), y.to(device)
        # y_hat = self(x)
        _, trajectory = self.model[:2](x) 
        y_hat = self.model[2:](trajectory[-1])
        acc = Accuracy()(y_hat, y)
        return {'test_loss': nn.CrossEntropyLoss()(y_hat, y), 'test_accuracy': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        self.log("test_loss", avg_loss)
        return {
            'avg_test_loss': avg_loss, 
            'avg_test_accuracy': avg_acc,
            'log': logs, 
            'progress_bar': logs}
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=5e-5)
        sched = {'scheduler': torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma = 9e-1),
                 'monitor': 'loss', 
                 'interval': 'step',
                 'frequency': 10 }
        return [opt], [sched]

    def train_dataloader(self):
        self.loader_len = len(self.trainloader)
        return self.trainloader

    def test_dataloader(self):
        self.test_loader_len = len(self.testloader)
        return self.testloader

class CIFARLearner(pl.LightningModule):
    def __init__(self, model:nn.Module, trainloader, testloader):
        super().__init__()
        self.model = model
        self.iters = 0.
        self.trainloader, self.testloader = trainloader, testloader
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        self.iters += 1.
        x, y = batch   
        self.model[2].vf.nfe = 0
        _, trajectory = self.model[:3](x) 
        y_hat = self.model[3:](trajectory[-1])   
        loss = nn.CrossEntropyLoss()(y_hat, y)
        # epoch_progress = self.iters / self.loader_len
        acc = Accuracy()(y_hat, y)
        nfe = self.model[2].vf.nfe ; self.model[2].vf.nfe = 0
        tqdm_dict = {
            'train_loss': loss, 
            'accuracy': acc, 
            'NFE': torch.tensor(nfe, dtype=torch.float)}
        self.log_dict({
            'epoch': self.current_epoch, 
            'train_loss': loss, 
            'accuracy': acc, 
            'NFE': torch.tensor(nfe, dtype=torch.float)
            }, on_epoch=True, on_step=False)
        return {'loss': loss, 'progress_bar': tqdm_dict}   

    def test_step(self, batch, batch_nb):
        x, y = batch
        _, trajectory = self.model[:3](x) 
        y_hat = self.model[3:](trajectory[-1])
        acc = Accuracy()(y_hat, y)
        return {'test_loss': nn.CrossEntropyLoss()(y_hat, y), 'test_accuracy': acc}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        logs = {'avg_test_loss': avg_loss, 'avg_test_accuracy': avg_acc}
        self.log_dict(logs)
        return logs
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=5e-4)
        sched = {'scheduler': torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma = 9e-1),
                 'monitor': 'loss', 
                 'interval': 'epoch'}
        return [opt], [sched]

    def train_dataloader(self):
        self.loader_len = len(self.trainloader)
        return self.trainloader

    def test_dataloader(self):
        self.test_loader_len = len(self.testloader)
        return self.testloader


class MetricTracker(Callback):
  def __init__(self):
    self.collection = []

  def on_train_epoch_end(self, trainer,*args, **kwargs):
    # logs = trainer.logged_metrics  
    logs = trainer.callback_metrics.copy()
    self.collection.append(logs) # track them
    # print(logs)

  def on_test_end(self, trainer, pl_module):
    logs = trainer.logged_metrics # access it here
    self.collection.append(logs)