import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_lr


class Model:
    def __init__(self, net, init_lr, batch_size, opt, train_set, test_set, gpu_id=0, decay=0, val_set=None):
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.train_set = train_set
        self.net = net.to(device=self.device)
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.train_set_size = len(train_set)
        self.test_set_size = len(test_set)
        self.have_val_set = val_set is not None
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                        batch_size=batch_size,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                       batch_size=batch_size,
                                                       shuffle=True)
        if self.have_val_set:
            self.val_set_size = len(val_set)
            self.val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                          batch_size=batch_size,
                                                          shuffle=True)

        if opt == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.init_lr, betas=(0.5, 0.999),
                                              weight_decay=decay)
        else:
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.init_lr, momentum=0.9, weight_decay=decay)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epoch=1, save_path=None, sampling_loader=None):
        self.net.train()
        lr = get_lr(init_lr=self.init_lr, epoch=epoch)
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        total_loss = 0
        acc_nums = 0
        if sampling_loader is not None:
            data_loader = sampling_loader
        else:
            data_loader = self.train_loader
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if sampling_loader is None:
                self.optimizer.zero_grad()
            outputs = self.net(inputs)

            predictions = outputs.argmax(-1)
            acc_nums += (predictions == labels).sum().item()

            loss = self.criterion(outputs, labels)
            loss.backward()
            if sampling_loader is None:
                self.optimizer.step()
            total_loss += loss.item()
        if save_path is not None:
            torch.save(self.net.state_dict(), save_path)
        return acc_nums / self.train_set_size, total_loss / self.train_set_size

    def test(self):
        return self.get_score(self.test_loader)

    def val(self):
        if not self.have_val_set:
            raise ValueError('No val dataset!')
        return self.get_score(self.val_loader)

    def get_score(self, loader):
        self.net.eval()
        total_loss = 0
        acc_nums = 0
        with torch.no_grad():
            for data_test in loader:
                images, labels = data_test
                images, labels = images.to(device=self.device), labels.to(device=self.device)
                output_test = self.net(images)

                predictions = output_test.argmax(-1)
                acc_nums += (predictions == labels).sum().item()

                loss = self.criterion(output_test, labels)
                total_loss += loss.item()
        return round(acc_nums / self.test_set_size, 4), total_loss / self.test_set_size

    def communicate_train(self, data, epoch, save_path=None):
        lr = get_lr(init_lr=self.init_lr, epoch=epoch)
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self.net.train()
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        predictions = outputs.argmax(-1)
        acc_nums = (predictions == labels).sum().item()
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return acc_nums / len(inputs), loss / len(inputs)

    def get_net(self):
        return self.net

    def change(self, epoch):
        lr = get_lr(init_lr=self.init_lr, epoch=epoch)
        for g in self.optimizer.param_groups:
            g['lr'] = lr


if __name__ == '__main__':
    pass
