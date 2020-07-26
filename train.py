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
from models.HGNN import GCNModel
from config import get_config
from datasets import load_feature_construct_H, randomedge_sample_H
from datasets import source_select
import argparse

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
    if cfg["activate_dataset"] == 'cora':
        H = hgut._edge_dict_to_H(edge_dict, cfg['percent'])  # 由
    for epoch in range(num_epochs):
        if cfg["activate_dataset"] == 'cora': \
                H = H
        # H = hgut._edge_dict_to_H(edge_dict, cfg['percent'])  # 由邻接表生成连接矩阵
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


def parsering():

    # Training settings
    parser = argparse.ArgumentParser()
    # Training parameter
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Disable validation during training.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=800,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Initial learning rate.')
    parser.add_argument('--lradjust', action='store_true',
                        default=False, help='Enable leraning rate adjust.(ReduceLROnPlateau or Linear Reduce)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--mixmode", action="store_true",
                        default=False, help="Enable CPU GPU mixing mode.")
    parser.add_argument("--warm_start", default="",
                        help="The model name to be loaded for warm start.")
    parser.add_argument('--debug', action='store_true',
                        default=False, help="Enable the detialed training output.")
    parser.add_argument('--dataset', default="cora", help="The data set")
    parser.add_argument('--datapath', default="data/", help="The data path.")
    parser.add_argument("--early_stopping", type=int,
                        default=0,
                        help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
    parser.add_argument("--no_tensorboard", action='store_true', default=False,
                        help="Disable writing logs to tensorboard")
    # add save_dir
    parser.add_argument('--save_dir', type=str, default="../results", help="The data path.")

    # Model parameter
    parser.add_argument('--type',default='gcnii',
                        help="Choose the model to be trained.(mutigcn, resgcn, densegcn, inceptiongcn)")
    parser.add_argument('--inputlayer', default='gcn',
                        help="The input layer of the model.")
    parser.add_argument('--outputlayer', default='gcn',
                        help="The output layer of the model.")
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--withbn', action='store_true', default=False,
                        help='Enable Bath Norm GCN')
    parser.add_argument('--withloop', action="store_true", default=False,
                        help="Enable loop layer GCN")
    parser.add_argument('--nhiddenlayer', type=int, default=1,
                        help='The number of hidden layers.')
    parser.add_argument("--normalization", default="AugNormAdj",
                        help="The normalization on the adj matrix.")
    parser.add_argument("--sampling_percent", type=float, default=1.0,
                        help="The percent of the preserve edges. If it equals 1, no sampling is done on adj matrix.")
    parser.add_argument("--pretrain_sampling_percent", type=float, default=1.0,
                        help="The percent of the preserve edges. If it equals 1, no sampling is done on adj matrix.")
    # parser.add_argument("--baseblock", default="res", help="The base building block (resgcn, densegcn, mutigcn, inceptiongcn).")
    parser.add_argument("--nbaseblocklayer", type=int, default=1,
                        help="The number of layers in each baseblock")  # same as '--layer' of gcnii
    parser.add_argument("--aggrmethod", default="default",
                        help="The aggrmethod for the layer aggreation. The options includes add and concat."
                             " Only valid in resgcn, densegcn and inecptiongcn")
    parser.add_argument("--task_type", default="full",
                        help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")
    # argument for gcnii
    parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--gpu', type=int, default=0, help='device id')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
    parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')  # gcnii*
    args = parser.parse_args()
    return args


def _main():
    print(f"Classification on {cfg['on_dataset']} dataset!!! class number: {n_class}")
    print(f"use MVCNN feature: {cfg['use_mvcnn_feature']}")
    print(f"use GVCNN feature: {cfg['use_gvcnn_feature']}")
    print(f"use MVCNN feature for structure: {cfg['use_mvcnn_feature_for_structure']}")
    print(f"use GVCNN feature for structure: {cfg['use_gvcnn_feature_for_structure']}")
    print('Configuration -> Start')
    pp.pprint(cfg)
    print('Configuration -> End')
    nhiddenlayer = 1
    args=parsering()
    args.nbaseblocklayer=cfg['nbaseblocklayer']
    args.hidden=cfg['n_hid']
    args.dropout=cfg['drop_out']
    args.lr = cfg['lr']
    # model_ft = HGNN(in_ch=fts.shape[1],
    #                 n_class=n_class,
    #                 n_hid=cfg['n_hid'],
    #                 dropout=cfg['drop_out'],
    #                 )  # 两层卷积
    # model_ft = MultiLayerHGNN(in_features=fts.shape[1], hidden_features=cfg['n_hid'], out_features=n_class,
    #                           nbaselayer=cfg['nbaseblocklayer'],
    #                           withbn=False, withloop=False, dropout=cfg['drop_out'],
    #                           aggrmethod="nores", dense=False)
    model_ft = GCNModel(nfeat=fts.shape[1],
                     nhid=args.hidden,
                     nclass=n_class,
                     nhidlayer=args.nhiddenlayer,
                     dropout=args.dropout,
                     baseblock=args.type,
                     inputlayer=args.inputlayer,
                     outputlayer=args.outputlayer,
                     nbaselayer=args.nbaseblocklayer,
                     activation=F.relu,
                     withbn=args.withbn,
                     withloop=args.withloop,
                     aggrmethod=args.aggrmethod,
                     mixmode=args.mixmode,
                     args=args)

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
