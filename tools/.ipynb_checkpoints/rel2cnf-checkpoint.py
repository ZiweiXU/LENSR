import os, sys
sys.path.append('..')
from itertools import combinations_with_replacement

from tqdm import tqdm

from model.Misc.Conversion import box_prop_name, POS_REL_NAMES_FULL, RelevantFormulaContainer


def relevant_clauses(clauses, object_pair_list, converter):
    """
    From a list of CNF clauses, pick out the clauses that have propositions related to objects in
    the give object list

    :param clauses: list, containing CNF clauses
    :param object_pair_list: list[list], containing pairs of object ids
    :param converter: model.Misc.Conversion.Converter object
    :return: list of clauses involved
    """

    def _relevant(c):
        for l in c:
            triple = converter.num2triple(abs(l))
            for obj_pair in object_pair_list:
                if obj_pair == [triple[1], triple[2]] or obj_pair == [triple[2], triple[1]]:
                    return True
        return False

    return [c for c in clauses if _relevant(c)]


def prepare_clauses(clauses, anno, converter, objs):
    this_obj = []
    # append object observation
    for a in anno:
        if a['object']['category'] not in this_obj:
            this_obj.append(a['object']['category'])
        if a['subject']['category'] not in this_obj:
            this_obj.append(a['subject']['category'])

    r_clauses = relevant_clauses(clauses,
                                 [
                                     list(j) for j in
                                     combinations_with_replacement(this_obj, 2)
                                 ],
                                 converter)
    assumptions = []
    for a in anno:
        assumptions.append([converter.name2num((
            box_prop_name(a['subject']['bbox'], a['object']['bbox']),
            objs[a['subject']['category']],
            objs[a['object']['category']]
        ))])

    r_clauses = r_clauses + assumptions
    r_container = RelevantFormulaContainer(r_clauses)

    return r_clauses, r_container


if __name__ == "__main__":
    import pickle as pk
    import json

    import pysat
    from pysat import formula

    from model.Misc.Conversion import box_prop_name, Converter, RelevantFormulaContainer, POS_REL_NAMES_FULL
    from model.Misc.Formula import cnf_to_dimacs, find

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

    img_filenames = []
    idx = 0
    
    if not os.path.exists('../dataset/VRD/vrd_raw/'):
        os.mkdir('../dataset/VRD/vrd_raw/')
    
    for key, anno in tqdm(annotation.items()):
        if len(anno) == 0:
            continue
        img_filenames.append(key)
        r_clauses, r_container = prepare_clauses(clauses, anno, converter, objs)
        num_atom = max([abs(j) for i in r_container.get_relevant_formula().clauses for j in i])
        cnf_to_dimacs(f'../dataset/VRD/vrd_raw/{idx}.cnf', r_container.get_relevant_formula().clauses, num_atom)
        r_sol_t, r_sol_f = find(r_container.get_relevant_formula(), 5, assumptions=[])
        for sol_t_idx in range(len(r_sol_t)):
            cnf_to_dimacs(f'../dataset/VRD/vrd_raw/{idx}.st{sol_t_idx}.cnf',
                          [[i] for i in r_sol_t[sol_t_idx]], num_atom)
            cnf_to_dimacs(f'../dataset/VRD/vrd_raw/{idx}.sf{sol_t_idx}.cnf',
                          [[i] for i in r_sol_f[sol_t_idx]], num_atom)
        idx += 1
    pk.dump(img_filenames, open(f'../dataset/VRD/vrd_raw/idx2filename.pk', 'wb'))
