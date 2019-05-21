import os, sys
sys.path.append('/home/xuziwei/CS6283/CS6283Project')
import json
import numpy as np
import pickle as pk

from pysat import formula
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

from model.Misc.Formula import dimacs_to_cnf
from model.Misc.Conversion import Converter, POS_REL_NAMES_FULL
from rel2cnf import prepare_clauses
"""
This script converts CNF generated from stanford VRD dataset to pygcn compatible data.
Note that it is different from cnf2data, because we have a 'relevant clause' concept when dealing with stanford vrd.
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


def write_var(var_list, features):
    relations_str = ''
    variables_str = ''

    variables_str += str(0) + '\t' + '\t'.join(list(map(str, features['Global']))) + '\t0\n'
    for i in range(len(var_list)):
        variables_str += str(i+1) + '\t' + '\t'.join(list(map(str, var_list[i]))) + '\t1\n'
    variables_str += str(len(var_list)+1) + '\t' + '\t'.join(list(map(str, features['And']))) + '\t3\n'

    for i in range(1, len(var_list) + 2):
        relations_str += str(i) + f'\t0\n'
    for i in range(1, len(var_list)+1):
        relations_str += str(i) + f'\t{str(len(var_list)+1)}\n'

    return variables_str, relations_str


def write_data(input_file, output_file, features):
    print(input_file)
    file_id = input_file.split('/')[-1].split('.')[-3] if '.s' in input_file else input_file.split('/')[-1].split('.')[-2]
    file_id = int(file_id)
    print(file_id)
    cnf, _ = dimacs_to_cnf(input_file)
    _, r_container = prepare_clauses(clauses, annotation[idx2filename[file_id]], converter, objs)
    cnf = [r_container.get_original_repr(c) for c in cnf]

    variables = open(output_file[0], 'w')
    relations = open(output_file[1], 'w')

    relations_str = ''
    variables_str = ''

    feature_OR = features['Or']
    feature_AND = features['And']
    feature_G = features['Global']


    # add global var
    feature = feature_G
    label = 0
    variables_str += str(0) + '\t'
    for j in range(len(feature)):
        variables_str += str(feature[j]) + '\t'
    variables_str += str(label) + '\n'

    # record known vars
    if '.s' not in input_file:
        known_var = {}
        var_id = 1
        and_vars = []
        or_vars = []
        or_children = {}
        for c in cnf:
            or_vars.append(var_id)
            or_children[var_id] = []
            variables_str += (str(var_id) + '\t' + '\t'.join(list(map(str, feature_OR))) + '\t2\n')
            current_or = var_id
            var_id += 1
            for l in c:
                if l not in known_var.keys():
                    known_var[l] = var_id
                    variables_str += (str(var_id) + '\t' + '\t'.join(list(map(str, _feature_leaf(int(l))))) + '\t1\n')
                    this_var_id = var_id
                    var_id += 1
                else:
                    this_var_id = known_var[l]
                or_children[current_or].append(this_var_id)
        and_vars.append(var_id)
        variables_str += (str(var_id) + '\t' + '\t'.join(list(map(str, feature_AND))) + '\t3\n')
        var_id += 1

        for i in range(1, var_id):
            relations_str += (str(i) + '\t' + '0\n')
        for or_var in or_vars:
            for or_child in or_children[or_var]:
                relations_str += (str(or_child) + '\t' + str(or_var) + '\n')
        for or_var in or_vars:
            relations_str += (str(or_var) + '\t' + str(and_vars[0]) + '\n')
    else:
        var_id = 1
        pseudo_clause = [i[0] for i in cnf]
        for l in pseudo_clause:
            variables_str += (str(var_id) + '\t' + '\t'.join(list(map(str, _feature_leaf(int(l))))) + '\t1\n')
            var_id += 1
        variables_str += (str(var_id) + '\t' + '\t'.join(list(map(str, feature_AND))) + '\t3\n')
        var_id += 1

        for i in range(1, var_id):
            relations_str += (str(i) + '\t' + '0\n')
        for i in range(1, var_id-1):
            relations_str += (str(i) + '\t' + str(var_id-1) + '\n')

    relations.write(relations_str)
    variables.write(variables_str)
    relations.close()
    variables.close()


if __name__ == "__main__":
    features = pk.load(open('../model/pygcn/pygcn/features.pk', 'rb'))['features']
    directory_in_str = '../dataset/VRD/vrd_raw/'
    directory_in_str_out = '../dataset/VRD/vrd/'
    
    if not os.path.exists(directory_in_str_out):
        os.mkdir(directory_in_str_out)

    def _worker(file):
        if file.endswith(".cnf"):
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
        for _ in tqdm(p.map(_worker, os.listdir(directory_in_str)), total=len(os.listdir(directory_in_str))):
            pass
