#!/bin/bash -l

conda activate GUIDE

python -m domainbed.scripts.sweep launch \
       --data_dir=domainbed/data \
       --output_dir=GUIDE_SD_TI \
       --command_launcher multi_gpu \
       --algorithms GUIDE  \
       --datasets TerraIncognita \
       --n_hparams 1 \
       --hparams '{"model_name": "stabilityai/stable-diffusion-2-1-base", "feature_model": "diffusion", "timestep": 50, "num_clusters": 5}'\
       --n_trials 1 \
       --steps 5001 \
       --skip_confirmation \