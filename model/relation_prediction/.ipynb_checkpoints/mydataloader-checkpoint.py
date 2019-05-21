from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import numpy as np
import pickle as pk


class VRD_dataset(Dataset):
    def __init__(self, train_set_keys, image_features_train, annotation_train, information):
        self.train_set_keys = train_set_keys
        self.image_features_train = image_features_train
        self.annotation_train = annotation_train
        self.information = information

    def __len__(self):
        return len(self.train_set_keys)

    def __getitem__(self, idx):
        img = self.train_set_keys[idx]

        pairs = list(self.annotation_train[img].keys())
        x = []
        y = []
        info = []
        for i in range(len(pairs)):
            key = pairs[i]
            relation = self.annotation_train[img][key]

            if relation == 100:
                if np.random.random() < 0.01 and (self.information[img][key][1][1] != self.information[img][key][2][1]):
                    x.append(self.image_features_train[img][key])
                    y.append(relation)
                    info.append(self.information[img][key])
            else:
                x.append(self.image_features_train[img][key])
                y.append(relation)
                info.append(self.information[img][key])
        x = Tensor(x)
        y = Tensor(y).long()
        # print ('debug',img,pairs,x,y,info)
        return x, y, info


class VRD_dataset_test(Dataset):
    def __init__(self, train_set_keys, image_features_train, annotation_train, information):
        self.train_set_keys = train_set_keys
        self.image_features_train = image_features_train
        self.annotation_train = annotation_train
        self.information = information

    def __len__(self):
        return len(self.train_set_keys)

    def __getitem__(self, idx):
        # print(idx)
        img = self.train_set_keys[idx]

        pairs = list(self.annotation_train[img].keys())
        x = []
        y = []
        info = []
        for i in range(len(pairs)):
            key = pairs[i]
            relation = self.annotation_train[img][key]

            if self.information[img][key][1][1] != self.information[img][key][2][1]:
                x.append(self.image_features_train[img][key])
                y.append(relation)
                info.append(self.information[img][key])
        x = Tensor(x)
        y = Tensor(y).long()
        # print ('debug',img,pairs,x,y,info)
        return x, y, info
