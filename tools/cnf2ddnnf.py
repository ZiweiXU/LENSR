import os, sys
sys.path.append('../')
import argparse

from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

from model.Misc.Formula import dimacs_to_nnf

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--ds_name', type=str, required=True)
args = parser.parse_args()
save_path = args.save_path
ds_name = args.ds_name

in_dir = f'{save_path}/{ds_name}_raw/'
out_dir = f'{save_path}/{ds_name}_ddnnf_raw/'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

def _worker(f):
    if f.endswith('.cnf'):
        output, r = dimacs_to_nnf(in_dir + f, out_dir + f[:-4] + '.nnf', '../c2d_linux')
        if 'nan' in output or r != 0:
            print(f'Failed {f}')


with Pool() as p:
    for _ in tqdm(p.uimap(_worker, os.listdir(in_dir)), total=len(os.listdir(in_dir))):
        pass
