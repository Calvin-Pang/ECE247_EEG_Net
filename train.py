from dataset.dataset import EEG_Dataset
import numpy as np
import os
import argparse
from model.eegnet import EEGNet, Multi_EEGNet, HR_EEGNet, ShallowConvNet, MSFNet
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, f1_score
import torch.nn as nn
import torch
import os
import yaml
import copy
import datetime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train(model, train_loader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for samples, labels in train_loader:
        samples, labels = samples.cuda(), labels.cuda()
        outputs = model(samples)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * samples.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    return epoch_loss, epoch_acc

def val(model, val_loader, loss_fn):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for samples, labels in val_loader:
            samples, labels = samples.cuda(), labels.cuda()
            outputs = model(samples)
            loss = loss_fn(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * samples.size(0)
            running_corrects += torch.sum(preds == labels.data)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    epoch_f1 = f1_score(y_true, y_pred, average='macro')
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)
    return epoch_loss, epoch_acc, epoch_f1


def main(model, train_loader, val_loader, test_loader, learning_rate, epoch, checkpoint_dir):
    now = datetime.datetime.now()
    model_name = model.__class__.__name__
    best_model = copy.deepcopy(model)
    best_acc = 0
    optimizer = Adam(model.parameters(), lr = learning_rate, weight_decay = 2e-5)
    loss_fn = nn.CrossEntropyLoss(reduction = 'mean')
    train_acc_log = []
    val_acc_log = []
    test_acc_log = []
    test_f1_log = []
    for i in range(epoch):
        epoch_id = i + 1
        print('epoch', epoch_id, 'Training begins...')
        train_loss, train_acc = train(model, train_loader, optimizer, loss_fn)
        train_acc_log .append(train_acc)
        print('train loss:', float(train_loss), 'train acc:', float(train_acc))

        val_loss, val_acc, val_f1 = val(model, val_loader, loss_fn)
        print('val loss:', float(val_loss), 'val acc:', float(val_acc))
        val_acc_log.append(val_acc)
        if val_acc > best_acc: 
            best_acc = val_acc
            best_model.load_state_dict(copy.deepcopy(model.state_dict()))

        _, test_acc, test_f1 = val(model, test_loader, loss_fn)
        print('test acc:', float(test_acc))
        test_acc_log.append(test_acc)
        test_f1_log.append(test_f1)
        print()
    checkpoint_name = os.path.join(checkpoint_dir, model_name + '_' + now.strftime('%m%d_%H_%M') + '.pt')
    torch.save(best_model.state_dict(), checkpoint_name)
    print('best checkpoint save:', checkpoint_name)
    best_epoch_id = val_acc_log.index(max(val_acc_log))
    print('best epoch:', best_epoch_id + 1)
    print('best val acc:', float(val_acc_log[best_epoch_id]))
    print('test acc:', float(test_acc_log[best_epoch_id]))
    print('test f1:', float(test_f1_log[best_epoch_id]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    torch.manual_seed(42)


    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print('Config loaded!')

    if config['model'] == 'HR_EEGNet': model = HR_EEGNet().cuda()
    elif config['model'] == 'EEGNet': model = EEGNet().cuda()
    elif config['model'] == 'Multi_EEGNet': model = Multi_EEGNet().cuda()
    elif config['model'] == 'ShallowConvNet': model = ShallowConvNet().cuda()
    elif config['model'] == 'MSFNet': model = MSFNet().cuda()
    print('model:', config['model'])
    
    train_dataset = EEG_Dataset(config['X_train'], config['y_train'])
    train_loader = DataLoader(dataset = train_dataset, batch_size = config['batch_size'], shuffle = True)

    val_dataset = EEG_Dataset(config['X_val'], config['y_val'])
    val_loader = DataLoader(dataset = val_dataset, batch_size = config['batch_size'])

    test_dataset = EEG_Dataset(config['X_test'], config['y_test'])
    test_loader = DataLoader(dataset = test_dataset, batch_size = config['batch_size'])


    main(model, train_loader, val_loader, test_loader, config['lr'], config['epoch'], config['checkpoint'])