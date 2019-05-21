from __future__ import division
from __future__ import print_function

import random
import time
import argparse
import os
import resource
import urllib.request
import json

import torch
import torch.multiprocessing
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import load_data, accuracy, RunningAvg, Mydataset
from models import GCN, MLP

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rlimit[1]))
torch.multiprocessing.set_sharing_strategy('file_system')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=50,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--indep_weights', action="store_true", default=False,
                    help='whether to use independent weights for different types of gates in ddnnf')
parser.add_argument('--dataset', type=str, required=True,
                    help='dataset')
parser.add_argument('--w_reg', type=float, default=0.1, help='strength of regularization')
parser.add_argument('--margin', type=float, default=1.0, help='margin in triplet margin loss')\

parser.add_argument('--dataloader_workers', type=int, default=5, help='number of workers for dataloaders')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# hyper parameters
# args.cuda = True
# indep_weights = False
# dataset = 'general'

indep_weights = args.indep_weights
dataset = args.dataset

# Load data
and_or = dataset in ['ddnnf', 'vrd_ddnnf']

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cuda_input(adj, features, labels, idx_train, idx_val, idx_test):
    if args.cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    return adj, features, labels, idx_train, idx_val, idx_test


file_list_raw = os.listdir(f'../data/{dataset}/')
file_list = set()
for file in file_list_raw:
    if '.s' not in file and '.and' not in file and '.or' not in file and '.rel' not in file:
        file_list.add(urllib.request.unquote(file))
file_list = list(file_list)

# split train test
# shuffle first
idx = np.arange(len(file_list))
np.random.seed(1)
np.random.shuffle(idx)
file_list = np.array(file_list)[idx]

split_idx = int(round(len(file_list) * 0.8, 0))
file_list_train = file_list[:split_idx]
file_list_test = file_list[split_idx:]
json.dump(list(file_list_test), open(dataset + '.testformula', 'w'), ensure_ascii=False)

dataset_train = Mydataset(dataset, file_list_train, and_or=and_or, args=args)
dataset_test = Mydataset(dataset, file_list_test, and_or=and_or, args=args)
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_workers)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=args.dataloader_workers)

print('Number of training samples:', len(dataset_train))
print('file list length: ', len(file_list), len(file_list_train), len(file_list_test))

adj0, features0, labels0, idx_train0, idx_val0, idx_test0, add_children0, or_children0 = load_data(
    '0', dataset, and_or=and_or)
model = GCN(nfeat=features0.shape[1],
            nhid=args.hidden,
            # nclass=labels.max().item() + 1,
            nclass=100,
            dropout=args.dropout,
            indep_weights=indep_weights)
mlp = MLP(dropout=args.dropout)
creterion = torch.nn.TripletMarginLoss(margin=args.margin, p=2)

def test():
    avg_loss = []
    avg_loss_CE = []
    total = 0
    correct = 0
    model.eval()
    mlp.eval()
    for _, data_test_group in tqdm(enumerate(dataloader_test), desc='Testing', total=len(dataloader_test)):
        # pass three times first, then back propagate the loss
        for data_test in data_test_group:
            vector3 = []
            for data_test_item in data_test:
                adj, features, labels, idx_train, idx_val, idx_test = cuda_input(*data_test_item[:-2])
                output = model(features.squeeze(0), adj.squeeze(0), labels)
                vector3.append(output[0])
            # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_test = creterion(vector3[0].unsqueeze(0), vector3[1].unsqueeze(0), vector3[2].unsqueeze(0))
            avg_loss.append(float(loss_test.cpu()))

            # back prop MLP
            input = torch.cat(
                (torch.cat((vector3[0], vector3[1])).unsqueeze(0), torch.cat((vector3[0], vector3[2])).unsqueeze(0)))
            pred = mlp(input)
            target = torch.LongTensor([1, 0]).cuda()
            mlp_loss = CE(pred, target)
            avg_loss_CE.append(float(mlp_loss.cpu()))

            # caclulate accuracy
            _, predicted = torch.max(pred.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = np.average(avg_loss)
    avg_loss_CE = np.average(avg_loss_CE)
    avg_acc = correct / total
    print('Test loss: {:.4f}'.format(avg_loss),
          'Test loss CE: {:.4f}'.format(avg_loss_CE),
          'Test Acc: {:.4f}'.format(avg_acc), )
    return avg_loss, avg_loss_CE, avg_acc