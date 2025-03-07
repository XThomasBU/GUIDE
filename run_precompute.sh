#!/bin/bash -l

module load miniconda

conda activate diff_domain_gen

python -m domainbed.scripts.precompute_feats --dataset_name "TerraIncognita" --model_name "stabilityai/stable-diffusion-2-1-base" --feature_model "diffusion" --timestep 50