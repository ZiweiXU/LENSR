import os, sys
sys.path.append('../')
import json
import pickle as pk

import numpy as np
from tqdm import tqdm
from pysat import formula
from pathos.multiprocessing import ProcessingPool as Pool

from model.Misc.Formula import dimacs_to_cnf
from model.Misc.Conversion import Converter, POS_REL_NAMES_FULL
from rel2cnf import prepare_clauses
"""
This script converts d-DNNF generated from stanford VRD dataset to pygcn compatible data.
Note that it is different from ddnnf2data, because we have a 'relevant clause' concept when dealing with stanford vrd.
"""

clauses = pk.load(open('../dataset/VRD/clauses.pk', 'rb'))
objs = pk.load(open('../dataset/VRD/objects.pk', 'rb'))
pres = pk.load(open('../dataset/VRD/predicates.pk', 'rb'))
annotation = json.load(open('../dataset/VRD/annotations_train.json'))
annotation_test = json.load(open('../dataset/VRD/annotations_test.json'))
word_vectors = pk.load(open('../dataset/VRD/word_vectors.pk', 'rb'))
tokenizers = pk.load(open('../dataset/VRD/tokenizers.pk', 'rb'))
variables = pk.load(open('../dataset/VRD/var_pool.pk', 'rb'))
var_pool = formula.IDPool(start_from=1)
for _, obj in variables['id2obj'].items():
    var_pool.id(obj)
converter = Converter(var_pool, pres, objs)
idx2filename = pk.load(open('../dataset/VRD/vrd_raw/idx2filename.pk', 'rb'))


def _feature_leaf(num):
    name = list(converter.num2name(abs(num)))
    for i in range(len(name)):
        if name[i] in POS_REL_NAMES_FULL.keys():
            name[i] = POS_REL_NAMES_FULL[name[i]]
    embedding = np.array(
        [word_vectors[tokenizers['vocab2token'][i]] for i in ' '.join([name[1], name[0], name[2]]).split(' ')])
    summed_embedding = np.sum(embedding, axis=0) / 3
    return summed_embedding if num > 0 else -summed_embedding


def write_data(input_file, output_file, features):
    ddnnf = open(input_file, 'r')
    variables = open(output_file[0], 'w')
    relations = open(output_file[1], 'w')

    and_children = open(output_file[2], 'w')
    or_children = open(output_file[3], 'w')

    relations_str = ''
    variables_str = ''

    and_children_list = []
    or_children_list = []

    num_feature = 50

    feature_OR = features['Or']
    feature_AND = features['And']
    feature_G = features['Global']

    file_id = input_file.split('/')[-1].split('.')[-3] if '.s' in input_file else input_file.split('/')[-1].split('.')[-2]
    file_id = int(file_id)
    cnf, _ = dimacs_to_cnf(input_file.replace('vrd_ddnnf_raw', 'vrd_raw').replace('nnf', 'cnf'))
    _, r_container = prepare_clauses(clauses, annotation[idx2filename[file_id]], converter, objs)
    cnf = [r_container.get_original_repr(c) for c in cnf]

    # add global var
    feature = feature_G
    label = 0
    variables_str += str(0) + '\t'
    for j in range(len(feature)):
        variables_str += str(feature[j]) + '\t'
    variables_str += str(label) + '\n'

    line_num = -1
    for line in ddnnf.readlines():
        if line_num > -1:
            line = line.split()
            type = line[0]
            children = line[1:]
            if type == 'L':
                feature = _feature_leaf(r_container.get_original_repr([int(children[0])])[0])
                label = 1  # leaf node
            elif type == 'O':
                feature = feature_OR
                label = 2  # OR node
                or_children_list.append([])
                for child in children[2:]:
                    child = int(child)
                    or_children_list[-1].append(child + 1)
                    relations_str += str(child + 1) + '\t' + str(line_num + 1) + '\n'
            elif type == 'A':
                feature = feature_AND
                label = 3  # AND node
                and_children_list.append([])
                for child in children[1:]:
                    child = int(child)
                    and_children_list[-1].append(child + 1)
                    relations_str += str(child + 1) + '\t' + str(line_num + 1) + '\n'

            variables_str += str(line_num + 1) + '\t'
            for j in range(len(feature)):
                variables_str += str(feature[j]) + '\t'
            variables_str += str(label) + '\n'

        line_num += 1

    # add edge for global variable
    for j in range(line_num):
        relations_str += str(j + 1) + '\t' + str(0) + '\n'

    relations.write(relations_str)
    variables.write(variables_str)
    relations.close()
    variables.close()

    json.dump(and_children_list, and_children)
    json.dump(or_children_list, or_children)



if __name__ == '__main__':
    features = pk.load(open('../model/pygcn/pygcn/features.pk', 'rb'))['features']
    directory_in_str = '../dataset/VRD/vrd_ddnnf_raw/'
    directory_in_str_out = '../dataset/VRD/vrd_ddnnf/'
    
    if not os.path.exists(directory_in_str_out):
        os.mkdir(directory_in_str_out)

    def _worker(file):
        if file.endswith(".nnf"):
            input_dire = os.path.join(directory_in_str, file)
            output_dire = [os.path.join(directory_in_str_out, file[:-4] + '.var'),
                           os.path.join(directory_in_str_out, file[:-4] + '.rel'),
                           os.path.join(directory_in_str_out, file[:-4] + '.and'),
                           os.path.join(directory_in_str_out, file[:-4] + '.or')]
            can_go_next = True
            for f in output_dire:
                can_go_next = can_go_next and os.path.exists(f)
            if can_go_next:
                return
            write_data(input_dire, output_dire, features)

    with Pool() as p:
        for _ in tqdm(p.imap(_worker, os.listdir(directory_in_str)), total=len(os.listdir(directory_in_str))):
            pass
