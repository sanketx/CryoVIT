#!/bin/bash

sample=$1
split_id=$2
model=$3
label_key=$4

env_dir=/tmp/$USER/"$(uuidgen)"
mkdir -p $env_dir
tar -xf ~/projects/libs/cryovit_env.tar -C $env_dir

$env_dir/cryovit_env/bin/python -m \
    cryovit.train_model \
    model=$model \
    label_key=$label_key \
    exp_name="fractional_sample_${model}_${label_key}" \
    dataset=fractional \
    dataset.sample=$sample \
    dataset.split_id=$split_id

rm -rf $env_dir