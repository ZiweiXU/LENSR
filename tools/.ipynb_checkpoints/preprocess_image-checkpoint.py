import pickle as pk

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from PIL import Image
import json
import sys
from torch import Tensor
from os import listdir
from os.path import isfile, join
from random import shuffle


class get_features(nn.Module):
    def __init__(self, ):
        super(get_features, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)

        self.resnet18_removed = list(self.resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*self.resnet18_removed)

    def forward(self, inputs):
        features = self.resnet18(inputs)
        return features


with open('../dataset/VRD/objects.json', 'r') as f:
    object_dic = json.load(f)
with open('../dataset/VRD/predicates.json', 'r') as f:
    relation_dic = json.load(f)
with open('../dataset/VRD/annotations_train.json', 'r') as f:
    annotation_train = json.load(f)
with open('../dataset/VRD/annotations_test.json', 'r') as f:
    annotation_test = json.load(f)

with open('../dataset/VRD/word_embed.json', 'r') as f:
    word_embed = json.load(f)

train_img_dire = '../dataset/VRD/sg_train_images/'
test_img_dire = '../dataset/VRD/sg_test_images/'
preprocess = transforms.Compose([transforms.Resize(size=(224, 224)), \
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])

train_imgs = [f for f in listdir(train_img_dire)]
test_imgs = [f for f in listdir(test_img_dire)]

model = get_features()

## data to write in file
preprocessed_annotation_train = {}
preprocessed_annotation_test = {}

preprocessed_image_features_train = {}
preprocessed_image_features_test = {}

info_train = {}
info_test = {}


def iterate_images(iterate_info, iterate_annotation_train, iterate_train_img_dire, iterate_train_imgs,
                   iterate_preprocessed_annotation_train, iterate_preprocessed_image_features_train):
    for train_image in range(len(iterate_train_imgs)):
        # for train_image in range(2):
        if train_image % 10 == 0:
            print('current number (train): ', train_image, len(iterate_train_imgs))
        img_name = iterate_train_imgs[train_image]
        img = Image.open(iterate_train_img_dire + img_name)

        iterate_preprocessed_annotation_train[img_name] = {}
        iterate_preprocessed_image_features_train[img_name] = {}
        iterate_info[img_name] = {}

        annotation_list = iterate_annotation_train[img_name]
        img_obj = set()
        relations_train_dic = {}
        for j in range(len(annotation_list)):
            annotation = annotation_list[j]

            label1 = annotation['subject']['category']
            label2 = annotation['object']['category']
            b1 = tuple(annotation['subject']['bbox'])  # ymin, ymax, xmin, xmax
            b2 = tuple(annotation['object']['bbox'])

            img_obj.add((label1, b1))
            img_obj.add((label2, b2))

            relation = annotation['predicate']

            relations_train_dic[((label1, b1), (label2, b2))] = relation

        img_obj = list(img_obj)
        for i in range(len(img_obj)):
            for j in range(len(img_obj)):
                if (img_obj[i], img_obj[j]) not in relations_train_dic:
                    relations_train_dic[(img_obj[i], img_obj[j])] = 70

        for item in relations_train_dic:
            ((label1, b1), (label2, b2)) = item
            relation = relations_train_dic[item]

            ymin = min(b1[0], b2[0])
            ymax = max(b1[1], b2[1])
            xmin = min(b1[2], b2[2])
            xmax = max(b1[3], b2[3])
            crop = transforms.functional.crop(img, i=ymin, j=xmin, h=ymax - ymin, w=xmax - xmin)
            box = preprocess(crop).unsqueeze(0)
            with torch.no_grad():
                features = model(box)
            features = features.view(features.numel()).numpy().tolist()

            label1_vec = list(np.mean(np.array(word_embed[object_dic[label1]]), axis=0))
            label2_vec = list(np.mean(np.array(word_embed[object_dic[label2]]), axis=0))

            l1_l2_f = label1_vec + label2_vec + features

            key = str(((label1, b1), (label2, b2)))

            iterate_preprocessed_annotation_train[img_name][key] = relation
            iterate_preprocessed_image_features_train[img_name][key] = l1_l2_f
            iterate_info[img_name][key] = (img_name, (label1, b1), (label2, b2))
    return iterate_preprocessed_annotation_train, iterate_preprocessed_image_features_train, iterate_info


preprocessed_annotation_train, preprocessed_image_features_train, info_train = \
    iterate_images(info_train, annotation_train, train_img_dire, train_imgs, preprocessed_annotation_train,
                   preprocessed_image_features_train)
preprocessed_annotation_test, preprocessed_image_features_test, info_test = \
    iterate_images(info_test, annotation_test, test_img_dire, test_imgs, preprocessed_annotation_test,
                   preprocessed_image_features_test)

with open('../dataset/VRD/preprocessed_annotation_train.pk', 'wb') as f:
    pk.dump(preprocessed_annotation_train, f)
with open('../dataset/VRD/preprocessed_image_features_train.pk', 'wb') as f:
    pk.dump(preprocessed_image_features_train, f)
with open('../dataset/VRD/preprocessed_annotation_test.pk', 'wb') as f:
    pk.dump(preprocessed_annotation_test, f)
with open('../dataset/VRD/preprocessed_image_features_test.pk', 'wb') as f:
    pk.dump(preprocessed_image_features_test, f)

pk.dump(info_train, open('../dataset/VRD/info_train.pk', 'wb'))
pk.dump(info_test, open('../dataset/VRD/info_test.pk', 'wb'))
