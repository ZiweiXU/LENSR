import json
import os, sys
import argparse
import pickle as pk
from itertools import combinations_with_replacement

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from PIL import Image

sys.path.append('../../')
sys.path.append('../pygcn/pygcn/')
from tqdm import tqdm
from pysat import formula

from torch import Tensor
from os import listdir
from os.path import isfile, join
from random import shuffle
from torch.utils.data import Dataset, DataLoader
from mydataloader import *
from model.Misc.Conversion import Converter, RelevantFormulaContainer, POS_REL_NAMES_FULL, box_prop_name
from model.pygcn.pygcn.utils import load_data
from model.pygcn.pygcn.models import GCN, MLP
from helper import augment_bbox

preprocessed_dire = '../../dataset/VRD/'
save_dire = './saved_model/'

with open(f'{preprocessed_dire}/preprocessed_annotation_train.pk', 'rb') as f:
    annotation_train = pk.load(f)
with open(f'{preprocessed_dire}/preprocessed_annotation_test.pk', 'rb') as f:
    annotation_test = pk.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', type=str)
parser.add_argument('--filename', type=str)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--logic-weight', type=float, default=0.1)
args = parser.parse_args()

preprocessed_annotation_train = pk.load(open(preprocessed_dire + 'preprocessed_annotation_train.pk', 'rb'))
preprocessed_image_features_train = pk.load(open(preprocessed_dire + 'preprocessed_image_features_train.pk', 'rb'))
preprocessed_annotation_test = pk.load(open(preprocessed_dire + 'preprocessed_annotation_test.pk', 'rb'))
preprocessed_image_features_test = pk.load(open(preprocessed_dire + 'preprocessed_image_features_test.pk', 'rb'))

info_train = pk.load(open(preprocessed_dire + 'info_train.pk', 'rb'))
info_test = pk.load(open(preprocessed_dire + 'info_test.pk', 'rb'))

# info_train = json.load(open(preprocessed_dire + 'info_train.pk', 'r'))
# info_test = json.load(open(preprocessed_dire + 'info_test.pk', 'r'))

clauses = pk.load(open(f'../../dataset/VRD/clauses.pk', 'rb'))
objs = pk.load(open(f'../../dataset/VRD/objects.pk', 'rb'))
pres = pk.load(open(f'../../dataset/VRD/predicates.pk', 'rb'))
word_vectors = pk.load(open(f'../../dataset/VRD/word_vectors.pk', 'rb'))
tokenizers = pk.load(open(f'../../dataset/VRD/tokenizers.pk', 'rb'))
variables = pk.load(open(f'../../dataset/VRD/var_pool.pk', 'rb'))
var_pool = formula.IDPool(start_from=1)
for _, obj in variables['id2obj'].items():
    var_pool.id(obj)
converter = Converter(var_pool, pres, objs)
idx2filename = pk.load(open('../../dataset/VRD/vrd_raw/idx2filename.pk', 'rb'))
node_features = pk.load(open('../../model/pygcn/pygcn/features.pk', 'rb'))['features']


def remove_tensor(info):
    info_clearn = []
    for pair in range(len(info)):
        info_clearn.append([])
        for iii in range(len(info[pair])):
            if iii == 0:
                info_clearn[-1].append(info[pair][iii][0])
            else:
                info_clearn[-1].append([])
                for jjj in range(len(info[pair][iii])):
                    if jjj == 0:
                        info_clearn[-1][-1].append(int(info[pair][iii][jjj][0]))
                    else:
                        info_clearn[-1][-1].append([])
                        for kkk in range(len(info[pair][iii][jjj])):
                            info_clearn[-1][-1][-1].append(int(info[pair][iii][jjj][kkk][0]))
    return info_clearn


class Relation_Pred(nn.Module):
    def __init__(self, MLP_hidden=30, num_relations=71):
        super(Relation_Pred, self).__init__()
        self.num_relations = num_relations

        self.num_features = 512
        self.num_labelvec = 300
        self.num_latent = 512


        self.MLP = nn.Sequential(nn.Linear(self.num_features + 2 * self.num_labelvec + 8, self.num_latent),
                                 nn.ReLU(),
                                 nn.Linear(self.num_latent, self.num_relations))

    def forward(self, inputs):

        prediction = self.MLP(inputs)
        return prediction


# Training
num_epoches = args.epochs
learning_rate = 0.001
batch_size = 1
logic_loss_weight = args.logic_weight

model = Relation_Pred().cuda()
criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_set_keys = list(preprocessed_annotation_train.keys())
test_set_keys = list(preprocessed_annotation_test.keys())

train_set = VRD_dataset(train_set_keys, preprocessed_image_features_train, preprocessed_annotation_train, info_train)
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = VRD_dataset_test(test_set_keys, preprocessed_image_features_test, preprocessed_annotation_test, info_test)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


# test dataset
def run_test(model_test, k=5):
    model_test.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        avg_loss = None
        for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            x, y, info = batch
            x = x.squeeze(0).cuda()
            y = y.squeeze(0).cuda()
            info = remove_tensor(info)

            x = augment_bbox(x, info)

            prediction = model_test(x)
            loss = F.nll_loss(F.log_softmax(prediction), y)
            if avg_loss is None:
                avg_loss = loss
            else:
                avg_loss += loss

            # _, predicted = torch.max(prediction.data, 1)
            _, predicted = torch.topk(prediction.data, k)
            pred = predicted.t()
            correct_num = pred.eq(y.view(1, -1).expand_as(pred))
            correct_k = correct_num[:k].view(-1).float().sum(0, keepdim=True)

            total += y.size(0)
            correct += correct_k[0]
        avg_loss = avg_loss / i
        acc = correct / total
        return avg_loss, acc


def relevant_clauses(clauses, object_pair_list, converter):
    """
    From a list of CNF clauses, pick out the clauses that have propositions related to objects in
    the give object list

    :param clauses: list, containing CNF clauses
    :param object_pair_list: list[list], containing pairs of object ids
    :param converter: model.Misc.Conversion.Converter object
    :return: list of clauses involved
    """

    def _relevant(c):
        for l in c:
            triple = converter.num2triple(abs(l))
            for obj_pair in object_pair_list:
                if obj_pair == [triple[1], triple[2]] or obj_pair == [triple[2], triple[1]]:
                    return True
        return False

    return [c for c in clauses if _relevant(c)]


def prepare_clauses(clauses, anno, converter, objs):
    this_obj = []
    # append object observation
    for a in anno:
        if a['object']['category'] not in this_obj:
            this_obj.append(a['object']['category'])
        if a['subject']['category'] not in this_obj:
            this_obj.append(a['subject']['category'])

    r_clauses = relevant_clauses(clauses,
                                 [
                                     list(j) for j in
                                     combinations_with_replacement(this_obj, 2)
                                 ],
                                 converter)
    assumptions = []
    for a in anno:
        assumptions.append([converter.name2num((
            box_prop_name(a['subject']['bbox'], a['object']['bbox']),
            objs[a['subject']['category']],
            objs[a['object']['category']]
        ))])

    r_clauses = r_clauses + assumptions
    r_container = RelevantFormulaContainer(r_clauses)

    return r_clauses, r_container


def assignment_to_gcn_compatible(var_list, node_features):
    """
    :param var_list:
    :param rel_list:
    :return: adj, feature, label
    """
    features = torch.stack(
        [torch.cuda.FloatTensor(node_features['Global'])] +
        var_list +
        [torch.cuda.FloatTensor(node_features['And'])])
    labels = torch.cuda.FloatTensor([0] + [1] * len(var_list) + [3])
    adj = torch.eye(len(labels), len(labels))
    adj[0, :] = 1
    adj[:, 0] = 1
    adj[-1, :] = 1
    adj[:, -1] = 1

    r_inv = adj.sum(dim=1) ** (-1)
    r_mav_inv = torch.diag(r_inv)
    adj_normalized = torch.mm(r_mav_inv, adj)

    return adj_normalized, features, labels


def get_formula_from_image(img_name, annotations, embedder, clauses, converter, objs, tokenizers):
    file_id = idx2filename.index(img_name)
    adj0, features0, labels0, idx_train0, idx_val0, idx_test0, _, _ = \
        load_data(filename=file_id, dataset=args.ds_name, override_path='../../dataset/VRD/', and_or=False)
    adj0 = adj0.to_dense().cuda()
    features0 = features0.cuda()
    labels0 = labels0.cuda()
    output = embedder(features0.squeeze(0), adj0.squeeze(0), labels0)
    return output[0]


def prediction_to_assignment_embedding(softmax_list, info_list, embedder, tokenizers, pres, objs):
    """
    :param softmax_list: a torch.Tensor
    :param info_list: [(img_name, (label1, bbox1), (label2, bbox2))]
    :param embedders: (pe, ce, ae)
    :return: the embedding of assignment
    """

    def _feature(name):
        embedding = np.array(
            [word_vectors[tokenizers['vocab2token'][i]] for i in name.split(' ')])
        summed_embedding = np.sum(embedding, axis=0)
        return summed_embedding

    embedded_clauses = []
    for idx, info in enumerate(info_list):
        prop = softmax_list[idx]
        sub = objs[info_list[idx][1][0]]
        obj = objs[info_list[idx][2][0]]

        e_p = 0
        for pres_idx in range(0, 70):
            e_p += prop[pres_idx] * torch.cuda.FloatTensor(_feature(tokenizers['token2vocab'][pres_idx]))
        e_p = e_p / 70
        e_p = (e_p + torch.cuda.FloatTensor(_feature(sub)) + torch.cuda.FloatTensor(_feature(obj))) / 3

        embedded_clauses.append(e_p)

    for idx, info in enumerate(info_list):
        pos = box_prop_name(info_list[idx][1][1], info_list[idx][2][1])
        if pos in POS_REL_NAMES_FULL.keys():
            pos = POS_REL_NAMES_FULL[pos]
        sub = objs[info_list[idx][1][0]]
        obj = objs[info_list[idx][2][0]]
        embedded_clauses.append((torch.cuda.FloatTensor(_feature(pos)) +
                                 torch.cuda.FloatTensor(_feature(sub)) +
                                 torch.cuda.FloatTensor(_feature(obj))) / 3)
    adj0, features0, labels0 = assignment_to_gcn_compatible(embedded_clauses, node_features)

    embedded_clauses = embedder(features0.cuda().squeeze(0), adj0.cuda().squeeze(0), labels0.cuda())

    return embedded_clauses


# load embedders
embedder_filename = args.filename
indep_weights = 'ind' in embedder_filename
embedder = torch.load('../pygcn/pygcn/model_save/' + embedder_filename)
# embedder = GCN(nfeat=50,
#                nhid=50,
#                # nclass=labels.max().item() + 1,
#                nclass=100,
#                dropout=0.5,
#                indep_weights=indep_weights).cuda()
# embedder.load_state_dict(torch.load('../pygcn/pygcn/model_save/' + embedder_filename))
# for p in embedder.parameters():
#     p.requires_grad = False


loss_save = {}
loss_save['train_avgloss_all'] = []
loss_save['train_avgloss_ce'] = []
loss_save['test_avgloss'] = []
loss_save['train_acc'] = []
loss_save['test_acc'] = []

model.train()
best_acc = 0
for iter in range(num_epoches):
    print('\n Iteration: ', iter)
    print(args.filename)
    correct = 0
    total = 0
    avg_loss_all = None
    avg_loss_ce = None
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        x, y, info = batch
        x = x.squeeze(0).cuda()
        y = y.squeeze(0).cuda()
        info = remove_tensor(info)

        x = augment_bbox(x, info)

        prediction = model(x)
        loss_entropy = F.nll_loss(F.log_softmax(prediction), y)

        # calculate loss
        formula_embedding = get_formula_from_image(info[0][0], annotation_train, embedder, clauses, converter, objs,
                                                   tokenizers)
        assignment_embedding = prediction_to_assignment_embedding(F.softmax(prediction), info, embedder, tokenizers,
                                                                  pres, objs)
        loss_embedding = (formula_embedding - assignment_embedding).norm()

        loss = loss_entropy + logic_loss_weight * loss_embedding

        # calcuate the avg loss and accuracy
        if avg_loss_all is None:
            avg_loss_all = loss
            avg_loss_ce = loss_entropy
        else:
            avg_loss_all += loss
            avg_loss_ce += loss_entropy

        _, predicted = torch.max(prediction.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss_all = avg_loss_all / i
    avg_loss_ce = avg_loss_ce / i
    acc = correct / total
    loss_save['train_avgloss_all'].append(float(avg_loss_all))
    loss_save['train_avgloss_ce'].append(float(avg_loss_ce))
    loss_save['train_acc'].append(float(acc))

    if not os.path.exists(save_dire):
        os.mkdir(save_dire)

    # test model for this epoch
    test_avg_loss, test_acc = run_test(model)
    if test_acc > best_acc:
        best_acc = test_acc
        # save model for this epoch
        torch.save(model, save_dire + "MLP_{}_{}_best".format(embedder_filename, str(logic_loss_weight)))
    torch.save(model, save_dire + "MLP_{}_{}_latest".format(embedder_filename, str(logic_loss_weight)))
    
    loss_save['test_avgloss'].append(float(test_avg_loss))
    loss_save['test_acc'].append(float(test_acc))
    print('Test Acc: ', test_acc)
    print('Test AvgLoss_CE: {:0.2f}'.format(test_avg_loss))

    # re-write the file
    json.dump(loss_save, open(save_dire + 'loss_save', 'w'), ensure_ascii=False)

# load model to do the testing
model_test = torch.load(save_dire + "MLP_{}_{}_best".format(embedder_filename, str(logic_loss_weight)))
model_test = model_test.cuda()
test_avg_loss, test_acc = run_test(model_test)
print('The final test avgloss: {:0.2f}; final test acc is: {:0.2f} '.format(test_avg_loss, test_acc))
