import os, sys, argparse
sys.path.append('../')
import numpy as np
import pickle as pk

from model.Misc.Formula import dimacs_to_cnf
from tqdm import tqdm

def write_data(input_file, output_file, features):

    cnf, _ = dimacs_to_cnf(input_file)

    variables = open(output_file[0], 'w')
    relations = open(output_file[1], 'w')

    relations_str = ''
    variables_str = ''

    and_children_list = []
    or_children_list = []

    num_feature = 50

    feature_OR = features['Or']
    feature_AND = features['And']
    feature_G = features['Global']
    # feature_leaf: [1,2,3]
    feature_leaf = {'1': features['a'], '2': features['b'],
                    '3': features['c'], '4': features['d'],
                    '5': features['e'], '6': features['f'],
                    '7': features['g'], '8': features['h'],
                    '9': features['i'], '10': features['j'],
                    '11': features['k'], '12': features['l']
                    }
    feature_leaf['-1'] = -feature_leaf['1']
    feature_leaf['-2'] = -feature_leaf['2']
    feature_leaf['-3'] = -feature_leaf['3']
    feature_leaf['-4'] = -feature_leaf['4']
    feature_leaf['-5'] = -feature_leaf['5']
    feature_leaf['-6'] = -feature_leaf['6']
    feature_leaf['-7'] = -feature_leaf['7']
    feature_leaf['-8'] = -feature_leaf['8']
    feature_leaf['-9'] = -feature_leaf['9']
    feature_leaf['-10'] = -feature_leaf['10']
    feature_leaf['-11'] = -feature_leaf['11']
    feature_leaf['-12'] = -feature_leaf['12']

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
                    variables_str += (str(var_id) + '\t' + '\t'.join(list(map(str, feature_leaf[str(l)]))) + '\t1\n')
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
            variables_str += (str(var_id) + '\t' + '\t'.join(list(map(str, feature_leaf[str(l)]))) + '\t1\n')
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
    # print(relations_str)

if __name__ == "__main__":
    features = pk.load(open('../model/pygcn/pygcn/features.pk', 'rb'))['features']
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--ds_name', type=str, required=True)
    args = parser.parse_args()
    ds_name = args.ds_name
    save_path = args.save_path

    directory_in_str = f'{save_path}/cnf_{ds_name}_raw/'
    directory_in_str_out = f'{save_path}/cnf_{ds_name}/'
    if not os.path.exists(directory_in_str_out):
        os.mkdir(directory_in_str_out)

    for file in tqdm(os.listdir(directory_in_str)):
        if file.endswith(".nnf"):
            input_dire = os.path.join(directory_in_str, file)
            output_dire = [os.path.join(directory_in_str_out, file[:-4] + '.var'),
                           os.path.join(directory_in_str_out, file[:-4] + '.rel'),
                           os.path.join(directory_in_str_out, file[:-4] + '.and'),
                           os.path.join(directory_in_str_out, file[:-4] + '.or')]

            write_data(input_dire, output_dire, features)
