import numpy as np
import scipy.sparse as sp
import torch
import json
import urllib.request

from torch.utils.data import Dataset


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(max(classes)+1)[c, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(filename=None, dataset=None, override_path=None, override_var=None, override_rel=None, and_or=True, directed=False):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))
    if not override_path:
        path = '../data/' + dataset + '/'
    else:
        path = override_path + '/' + dataset + '/'
    if not override_var:
        idx_features_labels = np.genfromtxt(f"{path}/{filename}.var",
                                        dtype=np.dtype(str))
    else:
        idx_features_labels = override_var

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    # print ("{}{}_relations".format(path, filename))
    # test = open("{}{}_and".format(path, filename))
    # print ('done')
    if not override_rel:
        edges_unordered = np.genfromtxt(f"{path}/{filename}.rel", dtype=np.int32)
    else:
        edges_unordered = override_rel
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    if not directed:
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)
    idx_train = range(0, len(idx))
    idx_val = range(0, len(idx))
    idx_test = range(0, len(idx))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    if and_or:
        and_children = json.load(open(f"{path}/{filename}.and"))
        or_children = json.load(open(f"{path}/{filename}.or"))
    else:
        and_children, or_children = [], []
    return adj, features, labels, idx_train, idx_val, idx_test, and_children, or_children


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


class Mydataset(Dataset):
    def __init__(self, dataset, dataset_split, and_or=True, ds_path='../../../dataset/Synthetic', args=None):
        super(Mydataset, self).__init__()
        self.dataset = dataset
        self.dataset_split = dataset_split
        self.and_or = and_or
        self.args = args
        self.ds_path = ds_path

    def __len__(self):
        return len(self.dataset_split)

    def __getitem__(self, item):
        data_item = []
        file_idx = self.dataset_split[item].split('.')[0]
        for item in range(5):
            try:
                adj0, features0, labels0, idx_train0, idx_val0, idx_test0, add_children0, or_children0 = self.load_data(
                    str(file_idx), self.dataset, and_or=self.and_or, directed=self.args.directed)
                adj1, features1, labels1, idx_train1, idx_val1, idx_test1, add_children1, or_children1 = self.load_data(
                    str(file_idx) + '.st' + str(item), self.dataset, and_or=self.and_or, directed=self.args.directed)
                adj2, features2, labels2, idx_train2, idx_val2, idx_test2, add_children2, or_children2 = self.load_data(
                    str(file_idx) + '.sf' + str(item), self.dataset, and_or=self.and_or, directed=self.args.directed)

                data_item.append([])  # main f, true assignment, false assignment

                data_item[-1].append(
                    (adj0.to_dense(), features0, labels0, idx_train0, idx_val0, idx_test0, add_children0, or_children0))

                data_item[-1].append(
                    (adj1.to_dense(), features1, labels1, idx_train1, idx_val1, idx_test1, add_children1, or_children1))

                data_item[-1].append(
                    (adj2.to_dense(), features2, labels2, idx_train2, idx_val2, idx_test2, add_children2, or_children2))
            except OSError as e:
                pass

        return data_item

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def load_data(self, filename, dataset, and_or=True, directed=True):
        """Load citation network dataset (cora only for now)"""
        # print('Loading {} dataset...'.format(dataset))
        path = self.ds_path + '/' + dataset + '/'
        idx_features_labels = np.genfromtxt(f"{path}/{filename}.var",
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = self.encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}

        edges_unordered = np.genfromtxt(f"{path}/{filename}.rel",
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        if not directed:
            # build symmetric adjacency matrix
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = self.normalize(adj + sp.eye(adj.shape[0]))

        idx_train = range(0, len(idx))
        idx_val = range(0, len(idx))
        idx_test = range(0, len(idx))

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        if and_or:
            and_children = json.load(open(f"{path}/{filename}.and"))
            or_children = json.load(open(f"{path}/{filename}.or"))
        else:
            and_children, or_children = [], []
        return adj, features, labels, idx_train, idx_val, idx_test, and_children, or_children

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(max(classes)+1)[c, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class RunningAvg:

    def __init__(self, window_size=10):
        self.window = np.zeros([window_size, 1])
        self.current_idx = 0
        self.window_size = window_size

    def add(self, value):
        self.window[self.current_idx] = value
        self.current_idx = (self.current_idx + 1) % self.window_size

    def avg(self):
        return self.window.mean()
