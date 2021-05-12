"""
    DCL or Independent training is performed by reading the config.ini configuration file, and the
    training log is automatically saved during the learning process.
"""

from models.model import Model
from utils import *
from distutils.util import strtobool
from net_loader import *
from data_loader import load_dataset
import os
import configparser
import torch
import platform

# Config
config = configparser.ConfigParser()
config.read('config.ini')
train_conf = config['train']

gpu_id = train_conf['gpu_id']
seed = int(train_conf['seed'])
net_type = train_conf['nn_type']
net_loader = get_loader(net_type)

pre_trained = strtobool(train_conf['pre_trained'])

log_base = '/data/Deep-Communication-Learning/' if platform.system() == 'Linux' else ''
save_path = log_base + 'logs/'
dataset = train_conf['dataset']
save_path += dataset

is_communicate = strtobool(train_conf['is_communicate'])
if is_communicate:
    save_path += '/communicate'
else:
    save_path += '/independent'

save_path += '/' + net_type
for i in range(1, 1000):
    end_save_path = '/result_' + str(i)
    if not os.path.exists(save_path + end_save_path):
        save_path += end_save_path
        os.makedirs(save_path)
        break

# back up training parameters to distinguish results
with open(save_path + '/config.ini', 'w') as f:
    config.write(f)

model_save_path = save_path + '/model.pth'
log_save_path = save_path + '/log.txt'

num_classes, train_set, test_set, in_channels, img_size = load_dataset(dataset, net_type)

epochs = int(train_conf['epochs'])

num_networks = int(train_conf['num_networks'])
lr_list = [float(train_conf['lr_init'])] * num_networks

batch_size_list = [int(train_conf['batch_size'])] * num_networks

decay_list = [float(train_conf['weight_decay'])] * num_networks

optimizer_list = [train_conf['optimizer']] * num_networks

beta = float(train_conf['beta'])

individually_train_batch_size = int(train_conf['individually_train_batch_size'])
print(f'Base save path: {save_path}')

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size_list[0], shuffle=True)


if __name__ == '__main__':
    seed_torch(seed=seed)
    model_list = [
        Model(net_loader(pre_trained, num_classes, in_channels=in_channels, img_size=img_size), init_lr, batch_size,
              opt, train_set, test_set, gpu_id=gpu_id, decay=decay)
        for init_lr, batch_size, opt, decay in zip(lr_list, batch_size_list, optimizer_list, decay_list)]
    t = TimeCount()
    for epoch in range(1, epochs + 1):
        test_loss_list = []
        test_acc_list = []
        for i, data in enumerate(train_loader):
            for model in model_list:
                model.communicate_train(data=data, epoch=epoch)

            if is_communicate and (i + 1) % individually_train_batch_size == 0:
                communicate_weight(model_list, beta)

        for i, model in enumerate(model_list):
            test_res = model.test()
            test_acc_list.append(test_res[0])
            test_loss_list.append(test_res[1])
        acc_max_idx = test_acc_list.index(max(test_acc_list))
        loss_min_idx = test_loss_list.index(min(test_loss_list))

        print_single(f'Epoch:{epoch}\nAcc_max_idx:{acc_max_idx}\nLoss_min_idx:{loss_min_idx}', log_save_path)
        print_single(f'test_acc:{test_acc_list}', log_save_path)
        print_single(f'Test_acc_Best: {max(test_acc_list)}', log_save_path)
        print_single(f'Use time: {t.count()} s', log_save_path)
        print_single('=' * 50, log_save_path)
