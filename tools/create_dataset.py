import sys
import json
import pickle as pk

import pysat

sys.path.append('..')
from model.Misc.Formula import find
from model.Misc.Conversion import Converter, POS_REL_NAMES


def gen_assumption(f, observations, converter):
    assumptions = []
    assert type(observations) is list

    objects_in_observation = list(set(
        [observation[1] for observation in observations] + [observation[2] for observation in
                                                            observations]))
    # objects_in_observation = [converter.objs.index(obs) for obs in objects_in_observation]

    for observation in observations:
        pos_observation, sub_observation, obj_observation = observation
        # existence constraint
        for obj in converter.objs:
            obj_exist = converter.name2num(('_exists', obj, obj))
            assumptions.append(
                obj_exist if obj in objects_in_observation else -obj_exist)
        # position mutually exclusive constraint
        for pos in POS_REL_NAMES:
            if pos != pos_observation:
                assumptions.append(-converter.name2num((pos, sub_observation, obj_observation)))

        assumptions.append(converter.name2num((pos_observation, sub_observation, obj_observation)))
    return assumptions


rels = pk.load(open('../resource/rels.pk', 'rb'))
pos_rels = rels['pos']

# load clauses, objects and predicates
clauses = pk.load(open('../resource/clauses.pk', 'rb'))
objs = pk.load(open('../resource/objects.pk', 'rb'))
pres = pk.load(open('../resource/predicates.pk', 'rb'))

# load all variables from pickled file
variables = pk.load(open('../resource/var_pool.pk', 'rb'))
var_pool = pysat.formula.IDPool(start_from=1)
for _, obj in variables['id2obj'].items():
    var_pool.id(obj)

converter = Converter(var_pool, pres, objs)

f = pysat.formula.CNF()
for c in clauses:
    f.append([i for i in c])

for pos_rel in pos_rels:
    observation = [('_n_right', 'umbrella', 'table'), ('_n_left', 'table', 'umbrella'),
                   ('_unique', 'table', 'table'), ('_unique', 'umbrella', 'umbrella')]
    assumptions = gen_assumption(f, observation, converter)

    true_sols, false_sols = find(f, 10, solver_name='Minisat22',
                                 assumptions=assumptions)
    for sol in true_sols:
        pos_sol = [i for i in sol if i > 0]
        print([converter.num2name(p_s) for p_s in pos_sol])

    break

with open('../resource/clauses.txt', 'w') as f:
    for c in clauses:
        f.write(str(c[0] / abs(c[0])) + str(converter.num2name(abs(c[0]))))
        f.write(' ')
        f.write(str(c[1] / abs(c[1])) + str(converter.num2name(abs(c[1]))))
        f.write('\n')
