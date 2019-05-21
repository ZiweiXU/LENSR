import os, argparse
import json
import pickle as pk

from tqdm import tqdm

def write_data(input_file, output_file, features):
    # features
    # literal: variable name with negation
    # and: 100
    # or: 200172.26.186.148
    # global (the first node): 300

    # input_file = 'test.txt.nnf'
    # output_file = ['variables','relations']
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

    line_num = -1
    for line in ddnnf.readlines():
        if line_num > -1:
            line = line.split()
            type = line[0]
            children = line[1:]
            if type == 'L':
                feature = feature_leaf[children[0]]
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
    # print(relations_str)


if __name__ == '__main__':
    features = pk.load(open('../model/pygcn/pygcn/features.pk', 'rb'))['features']
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--ds_name', type=str, required=True)
    args = parser.parse_args()
    ds_name = args.ds_name
    save_path = args.save_path

    directory_in_str = f'{save_path}/ddnnf_{ds_name}_raw/'
    directory_in_str_out = f'{save_path}/ddnnf_{ds_name}/'
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
