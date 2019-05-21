import numpy as np
import pickle as pk

preprocessed_annotation_train_raw = pk.load(open('../dataset/VRD/preprocessed_annotation_train.pk', 'rb'))
preprocessed_annotation_test_raw = pk.load(open('../dataset/VRD/preprocessed_annotation_test.pk', 'rb'))

clearn_train = {}
clearn_test = {}

img_train_list_raw = list(preprocessed_annotation_train_raw.keys())
for i in range(len(img_train_list_raw)):
    if preprocessed_annotation_train_raw[img_train_list_raw[i]] != {}:
        clearn_train[img_train_list_raw[i]] = preprocessed_annotation_train_raw[img_train_list_raw[i]]

img_test_list_raw = list(preprocessed_annotation_test_raw.keys())
for i in range(len(img_test_list_raw)):
    if preprocessed_annotation_test_raw[img_test_list_raw[i]] != {}:
        clearn_test[img_test_list_raw[i]] = preprocessed_annotation_test_raw[img_test_list_raw[i]]

print (len(clearn_train.keys()),len(clearn_test.keys()))
pk.dump(clearn_train, open('../dataset/VRD/preprocessed_annotation_train.pk', 'wb'))
pk.dump(clearn_test, open('../dataset/VRD/preprocessed_annotation_test.pk', 'wb'))