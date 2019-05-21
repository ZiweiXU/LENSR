#!/usr/bin/env bash
set -e
source activate LENSR

ds_names=("0303" "0306" "0606")

###### General form to CNF, d-DNNF #####
cd ../tools
for ds_name in ${ds_names[@]}; do
    python raw2data.py --save_path ../dataset/Synthetic/ --ds_name "${ds_name}"
done
cd -

###### CNF, d-DNNF to GCN compatible #####
cd ../tools
for ds_name in ${ds_names[@]}; do
    python cnf2data.py --save_path ../dataset/Synthetic/ --ds_name "${ds_name}"
    python ddnnf2data.py --save_path ../dataset/Synthetic/ --ds_name "${ds_name}"
done
cd -

###### Train Synthetic dataset #####
cd ../model/pygcn/pygcn

atom_options=("_0303" "_0306" "_0606" )
dataset_options=("general" "ddnnf" "cnf")

ind_options="--indep_weight"
reg_options="--w_reg 0.1"
non_reg_options="--w_reg 0.0"

for dataset in ${dataset_options[@]}; do
    for atom in "${atom_options[@]}"; do
        if [[ ${dataset} == 'general' ]]; then
            python train.py --ds_path ../../../dataset/Synthetic --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 5 ${non_reg_options} ${ind_options}
        fi
        if [[ ${dataset} == 'cnf' ]]; then
            python train.py --ds_path ../../../dataset/Synthetic --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 5 ${non_reg_options} ${ind_options}
            python train.py --ds_path ../../../dataset/Synthetic --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 5 ${non_reg_options}
        fi
        if [[ ${dataset} == 'ddnnf' ]]; then
            python train.py --ds_path ../../../dataset/Synthetic --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 5 ${reg_options} ${ind_options} 
            python train.py --ds_path ../../../dataset/Synthetic --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 5 ${reg_options}
            python train.py --ds_path ../../../dataset/Synthetic --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 5 ${ind_options}
            python train.py --ds_path ../../../dataset/Synthetic --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 5 ${non_reg_options}
        fi
    done
done
cd -
