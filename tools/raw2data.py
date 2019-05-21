import re, sys, os, argparse
sys.path.append('/home/xuziwei/CS6283/CS6283Project')
import argparse

import numpy as np
import pickle as pk

from tqdm import tqdm
from sympy import sympify, Symbol
from sympy.abc import A, B, C, D, E, F, G, H, I, J, K, L
from sympy.logic import And, Or, Not
from sympy.logic import to_cnf, to_nnf
from pysat import formula

from model.Misc.Formula import find, cnf_to_dimacs, dimacs_to_nnf

"""
This script generates three things: 
(1) the pygcn compatible data file for GENERAL form 
(2) CNF raw file
(3) DDNNF raw file
"""

def get_clauses(cnf_form):
    def _get_clauses(expr) -> tuple:
        if not isinstance(expr, And):
            return expr,
        return expr.args

    atoms = [None] + sorted(
        list(map(str, list(cnf_form.atoms()))),
        key=lambda x: [None, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'].index(x))
    clauses_tuple = _get_clauses(cnf_form)
    clauses = []
    for c in clauses_tuple:
        if type(c) == Not:
            clauses.append([-atoms.index(str(c)[1])])
        elif type(c) == Symbol:
            clauses.append([atoms.index(str(c))])
        elif type(c) == Or:
            clauses.append([atoms.index(str(i)) if str(i)[0] != '~' else -atoms.index(str(i)[1]) for i in c.args])
        else:
            raise ValueError(f'Unable to handle {str(c)}')

    return clauses, atoms


class pt_node:
    def __init__(self):
        self.args = None
        self.node_id = None
        self.data = None
        self.type = None

    def __repr__(self):
        return '(Node_id: ' + str(self.node_id) + '; Data: ' + str(self.data) + '; Type: ' + str(self.type) + ')'


def make_graph(fml, features, type_map):
    relations = []
    variables = []
    variables.append([0] + list(features['Global']) + [type_map['Global']])
    identified_var = [None]
    pt = pt_node()

    def build_var(root, pt):
        root_type = type(root)
        if root_type == Symbol:
            pt.data = root
            pt.type = root_type
            if str(root) not in identified_var:
                identified_var.append(str(root))
                pt.node_id = identified_var.index(str(root))
                variables.append([pt.node_id] + list(features[str(root)]) + [type_map['Symbol']])
            else:
                pt.node_id = identified_var.index(str(root))
            return
        elif root_type == Not:
            identified_var.append(str(root))
            pt.data = root
            pt.args = [pt_node() for i in range(len(root.args))]
            pt.node_id = len(identified_var) - 1
            pt.type = root_type
            variables.append([pt.node_id] + list(features['Not']) + [type_map['Not']])
        elif root_type == Or:
            identified_var.append(str(root))
            pt.data = root
            pt.args = [pt_node() for i in range(len(root.args))]
            pt.node_id = len(identified_var) - 1
            pt.type = root_type
            variables.append([pt.node_id] + list(features['Or']) + [type_map['Or']])
        elif root_type == And:
            identified_var.append(str(root))
            pt.data = root
            pt.args = [pt_node() for i in range(len(root.args))]
            pt.node_id = len(identified_var) - 1
            pt.type = root_type
            variables.append([pt.node_id] + list(features['And']) + [type_map['And']])

        if root_type in [Or, And, Not]:
            for i in range(len(root.args)):
                build_var(root.args[i], pt.args[i])

    def build_rel(pt):
        root_type = pt.type
        if root_type in [Symbol]:
            return
        if root_type in [Or, And, Not]:
            for c in pt.args:
                relations.append([c.node_id, pt.node_id])
        if root_type in [Or, And, Not]:
            for c in pt.args:
                build_rel(c)

    build_var(fml, pt)
    build_rel(pt)
    for i in range(1, len(variables)):
        relations.append([i, 0])
    return variables, relations, identified_var, pt


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--ds_name', type=str, required=True)
    args = parser.parse_args()
    
    d = pk.load(open('../model/pygcn/pygcn/features.pk', 'rb'))
    _, features, type_map = d['digit_to_sym'], d['features'], d['type_map']
    SAVE_PATH = args.save_path
    DIGIT_TO_SYM = [None, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    ds_name = args.ds_name

    if not os.path.exists(f'{SAVE_PATH}/cnf_{ds_name}_raw/'):
        os.mkdir(f'{SAVE_PATH}/cnf_{ds_name}_raw/')
    if not os.path.exists(f'{SAVE_PATH}/ddnnf_{ds_name}_raw/'):
        os.mkdir(f'{SAVE_PATH}/ddnnf_{ds_name}_raw/')
    if not os.path.exists(f'{SAVE_PATH}/general_{ds_name}/'):
        os.mkdir(f'{SAVE_PATH}/general_{ds_name}/')

    fml_strings = pk.load(open(f'{SAVE_PATH}formula_strings_{ds_name}.pk', 'rb'))
    for fml_idx in tqdm(range(len(fml_strings))):
        this_fml = sympify(fml_strings[fml_idx], evaluate=False)
        clauses, atom_mapping = get_clauses(to_cnf(this_fml))
        f = formula.CNF()
        for c in clauses: f.append(c)
        st, sf = find(f, 5, assumptions=[])
        if len(st) < 1:
            continue

        variables, relations, identified_var, pt = make_graph(this_fml, features=features, type_map=type_map)

        with open(f'{SAVE_PATH}/general_{ds_name}/{fml_idx}' + '.var', 'w') as f:
            for var in variables:
                f.write('\t'.join(list(map(str, var))))
                f.write('\n')
        with open(f'{SAVE_PATH}/general_{ds_name}/{fml_idx}' + '.rel', 'w') as f:
            for rel in relations:
                f.write('\t'.join(list(map(str, rel))))
                f.write('\n')

        # write cnf for fml
        cnf_to_dimacs(f'{SAVE_PATH}/cnf_{ds_name}_raw/{fml_idx}.cnf', clauses, len(atom_mapping) - 1)
        dimacs_to_nnf(f'{SAVE_PATH}/cnf_{ds_name}_raw/{fml_idx}.cnf',
                      f'{SAVE_PATH}/ddnnf_{ds_name}_raw/{fml_idx}.nnf',
                      '../../c2d_linux')

        for ii, tt in enumerate(st):
            tt_sym = ['~' + atom_mapping[abs(i)] if i < 0 else atom_mapping[abs(i)] for i in tt]
            tt_variables = []
            tt_relations = []
            tt_variables.append([0] + list(features['Global']) + [type_map['Global']])
            tt_variables.append([1] + list(features['And']) + [type_map['And']])
            idx = 2
            for tt_sym_i in tt_sym:
                if tt_sym_i[0] == '~':
                    tt_variables.append([idx] + list((-1) * features[tt_sym_i[1]]) + [type_map['Symbol']])
                else:
                    tt_variables.append([idx] + list(features[tt_sym_i]) + [type_map['Symbol']])
                tt_relations.append([idx, 1])
                idx += 1
            for i in range(1, idx):
                tt_relations.append([i, 0])
            with open(f'{SAVE_PATH}/general_{ds_name}/{fml_idx}.st{ii}.var', 'w') as f:
                for var in tt_variables:
                    f.write('\t'.join(list(map(str, var))))
                    f.write('\n')
            with open(f'{SAVE_PATH}/general_{ds_name}/{fml_idx}.st{ii}.rel', 'w') as f:
                for rel in tt_relations:
                    f.write('\t'.join(list(map(str, rel))))
                    f.write('\n')
            # create raw cnf file
            cnf_to_dimacs(f'{SAVE_PATH}/cnf_{ds_name}_raw/{fml_idx}.st{ii}.cnf', [[i] for i in tt], len(atom_mapping) - 1)
            dimacs_to_nnf(f'{SAVE_PATH}/cnf_{ds_name}_raw/{fml_idx}.st{ii}.cnf',
                          f'{SAVE_PATH}/ddnnf_{ds_name}_raw/{fml_idx}.st{ii}.nnf',
                          '../../c2d_linux')

        for ii, tt in enumerate(sf):
            tt_sym = ['~' + atom_mapping[abs(i)] if i < 0 else atom_mapping[abs(i)] for i in tt]
            tt_variables = []
            tt_relations = []
            tt_variables.append([0] + list(features['Global']) + [type_map['Global']])
            tt_variables.append([1] + list(features['And']) + [type_map['And']])
            idx = 2
            for tt_sym_i in tt_sym:
                if tt_sym_i[0] == '~':
                    tt_variables.append([idx] + list((-1) * features[tt_sym_i[1]]) + [type_map['Symbol']])
                else:
                    tt_variables.append([idx] + list(features[tt_sym_i]) + [type_map['Symbol']])
                tt_relations.append([idx, 1])
                idx += 1
            for i in range(1, idx):
                tt_relations.append([i, 0])
            with open(f'{SAVE_PATH}/general_{ds_name}/{fml_idx}.sf{ii}.var', 'w') as f:
                for var in tt_variables:
                    f.write('\t'.join(list(map(str, var))))
                    f.write('\n')
            with open(f'{SAVE_PATH}/general_{ds_name}/{fml_idx}.sf{ii}.rel', 'w') as f:
                for rel in tt_relations:
                    f.write('\t'.join(list(map(str, rel))))
                    f.write('\n')
            # create raw cnf file
            cnf_to_dimacs(f'{SAVE_PATH}/cnf_{ds_name}_raw/{fml_idx}.sf{ii}.cnf', [[i] for i in tt], len(atom_mapping) - 1)
            dimacs_to_nnf(f'{SAVE_PATH}/cnf_{ds_name}_raw/{fml_idx}.sf{ii}.cnf',
                          f'{SAVE_PATH}/ddnnf_{ds_name}_raw/{fml_idx}.sf{ii}.nnf',
                          '../../c2d_linux')
