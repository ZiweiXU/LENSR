import sys

sys.path.append('.')
sys.path.append('../')

import json
import pickle as pk
import copy
from model.Misc.Conversion import Converter

import pysat
from tqdm import tqdm

from model.Misc.Formula import find, find_truth
from model.Misc.Conversion import box_prop_name, POS_REL_NAMES, POS_REL_NAMES_PAIR

annotation = json.load(open('../dataset/VRD/annotations_train.json'))
objs = json.load(open('../dataset/VRD/objects.json'))
pres = json.load(open('../dataset/VRD/predicates.json'))

for i in POS_REL_NAMES:
    pres.append(i)
pres.append('_exists')
pres.append('_unique')

rel_to_pos = {}
# rel_to_pos_r = {}
# rel_to_impossible_pos = {}
clauses = []
var_pool = pysat.formula.IDPool(start_from=1)
converter = Converter(var_pool, pres, objs)

for _, annos in annotation.items():
    for anno in annos:
        pos_id = pres.index(
            box_prop_name(anno['subject']['bbox'], anno['object']['bbox']))
        pos_id_r = pres.index(
            box_prop_name(anno['object']['bbox'], anno['subject']['bbox']))
        rel_id = anno['predicate']
        subject_id = anno['subject']['category']
        object_id = anno['object']['category']

        if (rel_id, subject_id, object_id) not in rel_to_pos.keys():
            rel_to_pos[(rel_id, subject_id, object_id)] = []
        if (pos_id, subject_id, object_id) not in rel_to_pos[(rel_id, subject_id, object_id)]:
            rel_to_pos[(rel_id, subject_id, object_id)].append(
                (pos_id, subject_id, object_id))


for rel, positions in rel_to_pos.items():
    this_clause = [-var_pool.id(rel)]
    for pos in positions:
        this_clause.append(var_pool.id(pos))
    clauses.append(this_clause)

# postion equivalence constraint
for rel_pair in POS_REL_NAMES_PAIR:
    for sub in range(0, len(objs)):
        for obj in range(0, len(objs)):
            pos_var1 = var_pool.id((pres.index(rel_pair[0]), sub, obj))
            pos_var2 = var_pool.id((pres.index(rel_pair[1]), obj, sub))

            clauses.append([-pos_var1, pos_var2])
            clauses.append([-pos_var2, pos_var1])

pk.dump({'rel': list(rel_to_pos.keys())}, open('../dataset/VRD/rels.pk', 'wb'))
pk.dump(clauses, open('../dataset/VRD/clauses.pk', 'wb'))
pk.dump({'obj2id': dict(var_pool.obj2id), 'id2obj': dict(var_pool.id2obj)},
        open('../dataset/VRD/var_pool.pk', 'wb'))
pk.dump(pres, open('../dataset/VRD/predicates.pk', 'wb'))
pk.dump(objs, open('../dataset/VRD/objects.pk', 'wb'))
