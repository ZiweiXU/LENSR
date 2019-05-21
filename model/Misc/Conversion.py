import pysat

# in, out, o_left, o_right, o_up, o_down, n_left, n_right, n_up, n_down (left is xx the right obj)
POS_REL_NAMES = ['_out', '_in', '_o_left', '_o_right', '_o_up', '_o_down', '_n_left',
                 '_n_right', '_n_up', '_n_down']

POS_REL_NAMES_PAIR = [['_out', '_in'], ['_o_left', '_o_right'], ['_o_up', '_o_down'],
                      ['_n_left', '_n_right'], ['_n_up', '_n_down']]

POS_REL_NAMES_FULL = {'_out': 'outside', '_in': 'inside', '_o_left': 'overlapping left',
                      '_o_right': 'overlapping left', '_o_up': 'overlapping up',
                      '_o_down': 'overlapping down', '_n_left': 'not overlapping left',
                      '_n_right': 'not overlapping right', '_n_up': 'not overlapping up',
                      '_n_down': 'not overlapping down'}


# POS_REL_PAIRS = [['_out', '_in'], ['_o_left', '_o_right'], ['_o_up', '_o_down'],
#                  ['_n_left', '_n_right'], ['_n_up', '_n_down']]


def center(A):
    Aymin, Aymax, Axmin, Axmax = A
    return (Axmin + Axmax) / 2, (Aymin + Aymax) / 2


def box_prop(A, B):
    Aymin, Aymax, Axmin, Axmax = A
    Bymin, Bymax, Bxmin, Bxmax = B
    Axc, Ayc = center(A)
    Bxc, Byc = center(B)

    if Axmin <= Bxmax and Bxmin <= Axmax and Aymin <= Bymax and Bymin <= Aymax:  # if overlap
        if Axmin <= Bxmin and Axmax >= Bxmax and Aymin <= Bymin and Aymax >= Bymax:
            return 0
        elif Bxmin <= Axmin and Bxmax >= Axmax and Bymin <= Aymin and Bymax >= Aymax:
            return 1
        elif Axc <= Bxc:
            return 2
        elif Axc > Bxc:
            return 3
        elif Ayc <= Byc:
            return 4
        elif Ayc > Byc:
            return 5
    else:
        if Axc <= Bxc:
            return 6
        elif Axc > Bxc:
            return 7
        elif Ayc <= Byc:
            return 8
        elif Ayc > Byc:
            return 9


def box_prop_name(A, B):
    return POS_REL_NAMES[box_prop(A, B)]


def triple2num(triple, var_pool):
    return var_pool.id(triple)


def num2triple(num, var_pool):
    return var_pool.obj(num)


def num2name(num, var_pool, pres, objs):
    p, s, o = num2triple(num, var_pool)
    return pres[p], objs[s], objs[o]


def name2num(pre, sub, obj, pres, objs, var_pool):
    num_predicate = len(pres)
    p, s, o = pres.index(pre), objs.index(sub), objs.index(obj)
    return triple2num((p, s, o), var_pool)


class Converter:
    def __init__(self, var_pool, pres, objs):
        self.var_pool = var_pool
        self.pres = pres
        self.objs = objs

    def triple2num(self, triple):
        return self.var_pool.obj2id[triple]

    def name2num(self, triple):
        pre, sub, obj = triple
        p, s, o = self.pres.index(pre), self.objs.index(sub), self.objs.index(obj)
        return self.triple2num((p, s, o))

    def num2triple(self, num):
        return self.var_pool.id2obj[num]

    def num2name(self, num):
        pre, sub, obj = self.num2triple(num)
        return self.pres[pre], self.objs[sub], self.objs[obj]


class RelevantFormulaContainer:
    def __init__(self, r_clauses):
        self.r_clauses = r_clauses

        self.rvar_to_var = {}
        self.var_to_rvar = {}

        literal_counter = 1
        for r_c in r_clauses:
            for l in r_c:
                if abs(l) not in self.var_to_rvar.keys():
                    self.var_to_rvar[abs(l)] = literal_counter
                    self.rvar_to_var[literal_counter] = abs(l)
                    literal_counter += 1

    def get_relevant_formula(self):
        f = pysat.formula.CNF()
        for r_c in self.r_clauses:
            f.append([int(self.var_to_rvar[abs(l)] * abs(l) / l) for l in r_c])
        return f

    def get_original_repr(self, r_solution):
        return [int(self.rvar_to_var[abs(l)] * abs(l) / l) for l in r_solution]

def get_clauses_from_sympy(cnf_form):
    atoms = [None] + list(map(str,list(cnf_form.atoms())))
    clauses = []
    for clause_arg in cnf_form.args:
        this_clause = [atoms.index(str(i)) if str(i)[0] != '~' else -atoms.index(str(i)[1]) for i in clause_arg.args]
        clauses.append(this_clause)
    return clauses, atoms

def cnf_to_dimacs(file_name, clauses, num_atoms):
    with open(file_name,'w') as f:
        f.write(f'p cnf {num_atoms} {len(clauses)}\n')
        for c in clauses:
            for l in c:
                f.write(str(l) + ' ')
            f.write('0' + '\n')

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

def dimacs_to_nnf(file_name, c2d_path='./c2d_linux'):
    import os
    r = os.system(c2d_path + ' -in ' + file_name)
    return file_name + '.nnf', r
