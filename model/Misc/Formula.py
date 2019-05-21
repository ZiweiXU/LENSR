import subprocess

from tqdm import tqdm
import pysat
from pysat.formula import CNF, WCNF
from pysat.solvers import Solver
import numpy as np

from .Conversion import triple2num, num2triple, box_prop


def neg(a):
    out = []
    for i in range(len(a)):
        out.append(int(-a[i]))
    return out


def rand_assign(nv):
    out = []
    for i in range(nv):
        sign = np.random.choice([-1, 1])
        out.append(int((i + 1) * sign))
    return out


def find_truth(f, n, solver_name='g3', assumptions=None):
    out = []
    s = Solver(name=solver_name)
    s.append_formula(f.clauses)

    for idx, m in enumerate(s.enum_models(assumptions=assumptions)):
        out.append(m)
        if idx >= n - 1:
            break

    s.delete()
    return out

def find_false(f, n, max_try=100, solver_name='g3'):
    if n == 0:
        return []
    nv = f.nv
    out = []
    s = Solver(name=solver_name)
    s.append_formula(f.clauses)
    tries = 0
    while len(out) < n and tries < max_try:
        assign = rand_assign(nv)
        if s.solve(assumptions=assign) is False:
            out.append(assign)
        tries += 1
    s.delete()
    return out


def find_false_worse(f, n, solver_name='g3', sat_at_most=[]):
    assert n == len(sat_at_most)
    nv = f.nv
    out = []

    while len(out) < n:
        clause = []
        i = len(out)
        flips = np.random.rand(1,len(f.clauses))
        for idx, c in enumerate(f.clauses):
            if not flips[0][idx] >= sat_at_most[i]:
                clause.append(c)
            else:
                for v in c:
                    clause.append([-v])

        s = Solver(name=solver_name)
        s.append_formula(clause)
        for _, m in enumerate(s.enum_models()):
            out.append(m)
            break
        s.delete()
    return out


def count_sat_clauses(r_clauses, r_sol_f):
    sat_count = 0
    for clause in r_clauses:
        s = pysat.solvers.Solver()
        s.add_clause(clause)
        r = s.solve(assumptions=r_sol_f)
        sat_count = sat_count + 1 if r else sat_count
        s.delete()
    return sat_count / len(r_clauses)*1.0



def find(f, n, solver_name='g4', assumptions=None):
    truth = find_truth(f, n, solver_name=solver_name, assumptions=assumptions)
    false = find_false(f, len(truth), solver_name=solver_name)
    return truth, false


def find_worse(f, n, solver_name='g4', assumptions=None, sat_at_most=[]):
    truth = find_truth(f, n, solver_name=solver_name, assumptions=assumptions)
    false = find_false_worse(f, len(truth), solver_name=solver_name, sat_at_most=sat_at_most)
    return truth, false


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


def add_contraints(clauses, converter):
    """
    Given clauses, append existence constraints
    :param clauses: list of CNF clauses
    :param converter: converter: model.Misc.Conversion.Converter object
    :return: list of constraints
    """
    constraints = []
    raise NotImplementedError()


def cnf_to_dimacs(file_name, clauses, num_atoms):
    with open(file_name,'w') as f:
        f.write(f'p cnf {num_atoms} {len(clauses)}\n')
        for c in clauses:
            if type(c) != int:
                for l in c:
                    f.write(str(l) + ' ')
                f.write('0' + '\n')
            else:
                for cc in clauses:
                    f.write(str(cc) + ' ')
                f.write('0' + '\n')
                break


def dimacs_to_cnf(file_name):
    num_atoms = None
    clauses = []
    for line in open(file_name, 'r'):
        if line[0] == 'p':
            num_atoms = int(line.split()[-2])
        elif line[0] == 'c':
            continue
        else:
            l = list(map(int, line.split()[:-1]))
            clauses.append(l)
    return clauses, num_atoms


def dimacs_to_nnf(dimacs_path, nnf_path, c2d_path='./c2d_linux', ):
    import os
    r, output = subprocess.getstatusoutput(c2d_path + ' -in ' + dimacs_path)
    os.system('mv ' + dimacs_path+'.nnf' + ' ' + nnf_path)
    return output, r


if __name__ == "__main__":
    print(num2triple(triple2num(6, 53, 88)))

    print(box_prop([336, 489, 324, 458], [94, 175, 306, 590]))

    f1 = CNF()
    f1.append([-1, 2])

    print(f1.clauses)
    print(find(f1, 5))
