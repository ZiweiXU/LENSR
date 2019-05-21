import os
import argparse
import pickle as pk
import json
from functools import partial
import random

import numpy as np
import torch
from pysat.solvers import Solver
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

from utils import load_data
from models import GCN

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=None, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=50,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--use_old', '--use-old', action='store_true', default=False)
parser.add_argument('--bins', type=int, default=10)

parser.add_argument('--indep_weights', action='store_true', default=False)
parser.add_argument('--form', type=str, required=True)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--atoms', type=int, default=None)
parser.add_argument('--depth', type=int, default=None)
parser.add_argument('--w_reg', type=float, required=True)
parser.add_argument('--cls_reg', type=float, default=None)

args = parser.parse_args()
args.cuda = True

def set_randomness(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set_randomness(0)

num_sol_selected = 64
num_fml_selected = 1000

form = args.form
if args.dataset is None:
    args.dataset = f'_{args.atoms:02d}{args.depth:02d}'
dataset = args.dataset
if args.cls_reg is not None:
    cls_reg_name = '.cls'+str(args.cls_reg)
else:
    cls_reg_name = ''
if args.seed is not None:
    seed_name = '.seed' + str(args.seed)
else:
    seed_name = ''
num_vars = 2**args.atoms
ind_savename = '' if not args.indep_weights else '.ind'
reg = args.w_reg

filename = f'{form}{dataset}.reg{reg}{ind_savename}{cls_reg_name}{seed_name}'

# model_name = f'./model_save/{dataset}.model' if not args.indep_weights else f'./model_save/{dataset}.ind.model'
model_name = f'model_save/{filename}.model'
sol_data = f'sol_nnf{dataset}' if 'ddnnf' in form else f'sol_cnf{dataset}'
and_or = 'ddnnf' in form

print(f'Plot file will be saved as quality_plots/{filename}')

test_fmls = json.load(open(f'{form}{dataset}.testformula'))
all_fmls = [i for i in os.listdir(f'../data/{form}{dataset}/') if ('sf' not in i) and ('st' not in i) and ('var' in i)]
train_fmls = sorted(list(set(all_fmls) - set(test_fmls)))
random.seed(args.seed)
random.shuffle(train_fmls)
test_fmls = train_fmls[:num_fml_selected]
test_fmls = train_fmls
num_fmls = len(test_fmls)

if args.use_old:
    try:
        d = pk.load(open(f'./quality_plots/{filename}.dist_bin.quality.pk', 'rb'))
    except FileNotFoundError:
        d = pk.load(open(f'./quality_plots/{filename}.quality.pk', 'rb'))
    dist, sat = d['dist'], d['sat']
    fml_embeddings = d['fml_emb']
    sol_embeddings = d['sol_emb']
else:
    def count_sat_clauses(r_clauses, r_sol_f):
        sat_count = 0
        for clause in r_clauses:
            s = Solver()
            s.add_clause(clause)
            r = s.solve(assumptions=r_sol_f)
            sat_count = sat_count + 1 if r else sat_count
            s.delete()
        return sat_count / len(r_clauses) * 1.0


    def dimacs_to_cnf(file_name):
        num_atoms = None
        clauses = []
        for line in open(file_name, 'r'):
            if line[0] == 'p':
                num_atoms = int(line.split()[-2])
            elif line[0] == 'c':
                continue
            else:
                l = list(map(int, line.split()[:-1]))
                clauses.append(l)
        return clauses, num_atoms


    def cuda_input(adj, features, labels, idx_train, idx_val, idx_test):
        if args.cuda:
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()
        return adj, features, labels, idx_train, idx_val, idx_test


    adj0, features0, labels0, idx_train0, idx_val0, idx_test0, add_children0, or_children0 = load_data(
        f'{0}', form+dataset, and_or=and_or)
    # model = GCN(nfeat=features0.shape[1],
    #             nhid=args.hidden,
    #             # nclass=labels.max().item() + 1,
    #             nclass=100,
    #             dropout=args.dropout,
    #             indep_weights=args.indep_weights)
    # model = model.cuda()
    model = torch.load(model_name)
    model.cuda()
    model.eval()


    # sample solutions
    sol_list = list(range(num_vars))
    random.seed(args.seed)
    random.shuffle(sol_list)
    sol_list = sol_list[:num_sol_selected]
    num_vars = num_sol_selected

    # calculate embedding for solutions
    sol_embeddings = []
    for sol_idx in tqdm(sol_list, desc="Embedding solutions"):
        adj1, features1, labels1, idx_train1, idx_val1, idx_test1, add_children1, or_children1 = load_data(
            f'{sol_idx}', sol_data, and_or=and_or)
        adj1, features1, labels1, idx_train1, idx_val1, idx_test1 = cuda_input(adj1, features1, labels1,
                                                                               idx_train1, idx_val1,
                                                                               idx_test1)
        sol_embeddings.append(model(features1, adj1, labels1).detach().cpu().numpy()[0, :])

    # get solution as CNF formula
    sol_cnf = []
    for sol_idx in sol_list:
        file_sol = f'../data/sol_cnf{dataset}_raw/{sol_idx}.cnf'
        sol_cnf.append(dimacs_to_cnf(file_sol)[0])

    # calculate embedding for formulae
    fml_embeddings = []
    for formula_idx in tqdm(range(num_fmls), desc="Embedding formula"):
        adj0, features0, labels0, idx_train0, idx_val0, idx_test0, add_children0, or_children0 = load_data(
            f'{test_fmls[formula_idx].split(".")[0]}',
            form+dataset, and_or=and_or)
        adj0, features0, labels0, idx_train0, idx_val0, idx_test0 = cuda_input(adj0, features0, labels0,
                                                                               idx_train0, idx_val0,
                                                                               idx_test0)
        fml_embeddings.append(model(features0, adj0, labels0).detach().cpu().numpy()[0, :])

    # get formulae as CNF formula
    fml_cnf = []
    for formula_idx in range(num_fmls):
        # file_cnf = f'../../../../generated_cnfs/{formula_idx}.cnf'
        file_cnf = f'../data/cnf_new_raw/{test_fmls[formula_idx].split(".")[0]}.cnf'
        fml_cnf.append(dimacs_to_cnf(file_cnf)[0])


    def _distance(x, y_vec, num_fmls):
        dist = np.zeros(num_fmls)
        for j in range(num_fmls):
            dist[j] = np.linalg.norm(x - y_vec[j])
        return dist

    def _satisfy(x, y, num_fmls):
        sat = np.zeros(num_fmls)
        for j in range(num_fmls):
            sat[j] = count_sat_clauses(y[j], [l[0] for l in x])
        return sat

    # this uses multiple processes to speed up computation
    if 2 ** args.atoms > 256:
        print('Calculating pairwise distance...')
        with Pool() as p:
            dist = list(tqdm(p.imap(partial(_distance, y_vec=fml_embeddings, num_fmls=num_fmls), sol_embeddings), total=len(sol_embeddings)))
            sat = list(tqdm(p.imap(partial(_satisfy, y=fml_cnf, num_fmls=num_fmls), sol_cnf), total=len(sol_cnf)))
    else:
        dist = np.zeros((num_vars, num_fmls))
        sat = np.zeros((num_vars, num_fmls))
        for i in tqdm(range(num_vars), desc="Pairwise distance"):
            for j in range(num_fmls):
                dist[i][j] = np.linalg.norm(sol_embeddings[i] - fml_embeddings[j])
                sat[i][j] = count_sat_clauses(fml_cnf[j], [l[0] for l in sol_cnf[i]])

    dist = np.array(dist)
    sat = np.array(sat)

def normalize(a):
    min_a = min(a)
    max_a = max(a)
    e = 1e-6
    a = [(a[i] - min_a)/(max_a-min_a+e) for i in range(len(a))]
    return a

sat_flatten = sat.flatten()
dist_flatten = dist.flatten()
# sats_sorted = sorted(list(set(list(sat_flatten))))
# dists_sorted = sorted(list(set(list(dist_flatten))))
sats_sorted = sorted(list(sat_flatten))
dists_sorted = sorted(list(dist_flatten))


####### plot according to sat bins
sat_bins = np.linspace(sats_sorted[0], sats_sorted[-1], args.bins)
sat_bins_range = [(sat_bins[k], sat_bins[k + 1]) for k in range(len(sat_bins) - 1)]
dists = [[] for _ in range(len(sat_bins_range))]
for i in range(num_vars * num_fmls):
    range_slot = sat_bins_range.index(list(filter(lambda x: x[0] <= sat_flatten[i] <= x[1], sat_bins_range))[0])
    dists[range_slot].append(dist_flatten[i])
# dists = [[] for _ in range(len(sats))]
# for i in range(num_vars * num_fmls):
#     dists[sats.index(sat_flatten[i])].append(dist_flatten[i])

dists_count = []
for i in range(len(sat_bins) - 1):
    dists_count.append(len(dists[i]))
    dists[i] = np.mean(dists[i])
#     dists[i] = np.median(dists[i])
print ('dists_count (sats as bins)', dists_count)

pk.dump(
    {'dist': dist, 'sat': sat, 'fml_emb': fml_embeddings, 'sol_emb': sol_embeddings, 'dists': dists, 'sats_sorted': sats_sorted,
     'sat_bins': sat_bins},
    open(f'./quality_plots/{filename}.sat_bin.quality.pk', 'wb'))

f = plt.figure()
plt.scatter(x=dists, y=sat_bins[:-1])
plt.xlabel('Distance')
plt.ylabel('Satisfied Clauses')
plt.title(f'Quality Plot of {filename}, corr={np.corrcoef(dists, sat_bins[:-1])[0][1]}')
plt.show()
f.savefig(f'./quality_plots/{filename}.sat_bin.quality.pdf')
print(f'Plot file saved as quality_plots/{filename}.sat_bin.quality')


# ######## plot according to distance bins
# dist_bins = []
# num_points_bin = int(np.ceil(len(dists_sorted)/(args.bins-1)))
# print ('numpoints',len(dists_sorted),args.bins)
# for i in range(args.bins):
#     if i*num_points_bin >= len(dists_sorted):
#         idx = -1
#     else:
#         idx = i*num_points_bin
#     dist_bins.append(dists_sorted[idx])
# dist_bins[-1] = dist_bins[-1]+0.0001
# dist_bins = np.array(dist_bins)

# # dist_bins = np.linspace(dists_sorted[0], dists_sorted[-1], args.bins)
# dist_bins_range = [(dist_bins[k], dist_bins[k + 1]) for k in range(len(dist_bins) - 1)]
# sats = [[] for _ in range(len(dist_bins_range))]
# for i in range(num_vars * num_fmls):
#     range_slot = dist_bins_range.index(list(filter(lambda x: x[0] <= dist_flatten[i] <= x[1], dist_bins_range))[0])
#     sats[range_slot].append(sat_flatten[i])
# sats_count = []
# for i in range(len(dist_bins) - 1):
#     sats_count.append(len(sats[i]))
#     sats[i] = np.mean(sats[i])
# print ('sats_count (dists as bins)', sats_count)

# pk.dump(
#     {'dist': dist, 'sat': sat, 'fml_emb': fml_embeddings, 'sol_emb': sol_embeddings, 'dists': sats, 'sats_sorted': dists_sorted,
#      'sat_bins': dist_bins},
#     open(f'./quality_plots/{filename}.dist_bin.quality.pk', 'wb'))

# f = plt.figure()
# # plt.scatter(x=sats, y=dist_bins[:-1])
# plt.scatter(x=dist_bins[:-1],y=sats)
# plt.xlabel('Distance')
# plt.ylabel('Satisfied Clauses')
# plt.title(f'Quality Plot of {filename}, corr={np.corrcoef(sats, dist_bins[:-1])[0][1]}')
# plt.show()
# f.savefig(f'./quality_plots/{filename}.dist_bin.quality.pdf')
# print(f'Plot file saved as quality_plots/{filename}.dist_bin.quality')

# ####### Adaptive bins
# sat_bins = []
# num_points_bin = int(np.ceil(len(sats_sorted)/(args.bins-1)))

# for i in range(args.bins):
#     if i*num_points_bin >= len(sats_sorted):
#         idx = -1
#     else:
#         idx = i*num_points_bin
#     sat_bins.append(sats_sorted[idx])
# sat_bins = np.array(sat_bins)
# # sat_bins = np.linspace(sats_sorted[0], sats_sorted[-1], args.bins)
# sat_bins_range = [(sat_bins[k], sat_bins[k + 1]) for k in range(len(sat_bins) - 1)]

# dists = [[] for _ in range(len(sat_bins_range))]
# for i in range(num_vars * num_fmls):
#     range_slot = sat_bins_range.index(list(filter(lambda x: x[0] <= sat_flatten[i] <= x[1], sat_bins_range))[0])
#     dists[range_slot].append(dist_flatten[i])
# # dists = [[] for _ in range(len(sats))]
# # for i in range(num_vars * num_fmls):
# #     dists[sats.index(sat_flatten[i])].append(dist_flatten[i])

# dists_count = []
# for i in range(len(sat_bins) - 1):
#     dists_count.append(len(dists[i]))
#     dists[i] = np.mean(dists[i])
# #     dists[i] = np.median(dists[i])
# print ('dists_count (sats as bins)', dists_count)

# pk.dump(
#     {'dist': dist, 'sat': sat, 'fml_emb': fml_embeddings, 'sol_emb': sol_embeddings, 'dists': dists, 'sats_sorted': sats_sorted,
#      'sat_bins': sat_bins},
#     open(f'./quality_plots/{filename}.sat_bin.quality.pk', 'wb'))

# f = plt.figure()
# plt.scatter(x=dists, y=sat_bins[:-1])
# plt.xlabel('Distance')
# plt.ylabel('Satisfied Clauses')
# plt.title(f'Quality Plot of {filename}, corr={np.corrcoef(dists, sat_bins[:-1])[0][1]}')
# plt.show()
# f.savefig(f'./quality_plots/{filename}.sat_bin.quality.pdf')
# print(f'Plot file saved as quality_plots/{filename}.sat_bin.quality')