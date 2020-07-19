import os
import time
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pprint as pp
import utils.hypergraph_utils as hgut
from models import HGNN, MultiLayerHGNN
from config import get_config
from datasets import load_feature_construct_H, randomedge_sample_H
from datasets import source_select

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cfg = get_config('config/config.yaml')

# initialize data
data_dir = cfg['modelnet40_ft'] if cfg['on_dataset'] == 'ModelNet40' \
    else cfg['ntu2012_ft']
if cfg["activate_dataset"] == 'cora':
    source = source_select(cfg)
    print(f'Using {cfg["activate_dataset"]} dataset')
    fts, lbls, idx_train, idx_val, idx_test, n_category, adj, edge_dict = source(cfg)
else:
    fts, lbls, idx_train, idx_test, mvcnn_dist, gvcnn_dist = \
        load_feature_construct_H(data_dir,
                                 m_prob=cfg['m_prob'],
                                 K_neigs=cfg['K_neigs'],
                                 is_probH=cfg['is_probH'],
                                 use_mvcnn_feature=cfg['use_mvcnn_feature'],
                                 use_gvcnn_feature=cfg['use_gvcnn_feature'],
                                 use_mvcnn_feature_for_structure=cfg['use_mvcnn_feature_for_structure'],
                                 use_gvcnn_feature_for_structure=cfg[
                                     'use_gvcnn_feature_for_structure'])  # TODO: 放进epoch随机构建 看论文DHSL

# G = hgut.generate_G_from_H(H) # D_v^1/2 H W D_e^-1 H.T D_v^-1/2 :
n_class = int(lbls.max()) + 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# transform data to device
fts = torch.Tensor(fts).to(device)  # features -> fts
lbls = torch.Tensor(lbls).squeeze().long().to(device)
# G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500):
    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if cfg['tensorboard']:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(
            os.path.join(cfg['result_root'], f'{cfg["nbaseblocklayer"]}-layers_{cfg["percent"]}-percent-{time_start}'))
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if cfg["activate_dataset"] == 'cora':
            H = hgut._edge_dict_to_H(edge_dict, cfg['percent'])  # 由邻接表生成连接矩阵
            # H = hgut.adj_to_H(adj, cfg['percent'])
        else:
            H = randomedge_sample_H(mvcnn_dist=mvcnn_dist,
                                    gvcnn_dist=gvcnn_dist,
                                    split_diff_scale=False,
                                    m_prob=cfg['m_prob'],
                                    K_neigs=cfg['K_neigs'],
                                    is_probH=cfg['is_probH'],
                                    percent=cfg['percent'],
                                    use_mvcnn_feature_for_structure=cfg['use_mvcnn_feature_for_structure'],
                                    use_gvcnn_feature_for_structure=cfg['use_gvcnn_feature_for_structure'])
        G = hgut.generate_G_from_H(H)  # D_v^1/2 H W D_e^-1 H.T D_v^-1/2 :
        G = torch.Tensor(G).to(device)
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_val  # idx_test

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, G)
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            # statistics
            running_loss += loss.item() * fts.size(0)
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if cfg['tensorboard']:
                if phase == 'train':
                    writer.add_scalar('loss/train', loss, epoch)
                    writer.add_scalar('acc/train', epoch_acc, epoch)
                else:
                    writer.add_scalar('loss/val', loss, epoch)
                    writer.add_scalar('acc/val', epoch_acc, epoch)
                    writer.add_scalar('best_acc/val', best_acc, epoch)

        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:4f}')
            print('-' * 20)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    # test
    outputs = model(fts, G)
    # loss = criterion(outputs[idx_test], lbls[idx_test])
    _, preds = torch.max(outputs, 1)
    test_acc = torch.sum(preds[idx_test] == lbls.data[idx_test]).double() / len(idx_test)
    print(f"test_accuracy: {test_acc}")
    if cfg['tensorboard']:
        writer.add_histogram('best_acc', test_acc)
    return model


def _main():
    print(f"Classification on {cfg['on_dataset']} dataset!!! class number: {n_class}")
    print(f"use MVCNN feature: {cfg['use_mvcnn_feature']}")
    print(f"use GVCNN feature: {cfg['use_gvcnn_feature']}")
    print(f"use MVCNN feature for structure: {cfg['use_mvcnn_feature_for_structure']}")
    print(f"use GVCNN feature for structure: {cfg['use_gvcnn_feature_for_structure']}")
    print('Configuration -> Start')
    pp.pprint(cfg)
    print('Configuration -> End')

    # model_ft = HGNN(in_ch=fts.shape[1],
    #                 n_class=n_class,
    #                 n_hid=cfg['n_hid'],
    #                 dropout=cfg['drop_out'],
    #                 )  # 两层卷积
    model_ft = MultiLayerHGNN(in_features=fts.shape[1], hidden_features=cfg['n_hid'], out_features=n_class,
                              nbaselayer=cfg['nbaseblocklayer'],
                              withbn=False, withloop=False, dropout=cfg['drop_out'],
                              aggrmethod="nores", dense=False)
    print(model_ft)
    model_ft = model_ft.to(device)

    optimizer = optim.Adam(model_ft.parameters(), lr=cfg['lr'],
                           weight_decay=cfg['weight_decay'])
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=cfg['weight_decay)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg['milestones'],
                                               gamma=cfg['gamma'])
    criterion = torch.nn.CrossEntropyLoss()

    model_ft = train_model(model_ft, criterion, optimizer, schedular, cfg['max_epoch'], print_freq=cfg['print_freq'])


if __name__ == '__main__':
    seed_num = 1000

    setup_seed(seed_num)
    _main()
