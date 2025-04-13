# PyTorch Code for "What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization"

## Why Diffusion Features and why do we need it?
- Provides rich domain-specific information to the classifier to aid in classification to new unseen domains.


![Diffusion Latent Space](assets/fig_latent_space.png)
![Class vs Domain NMI](assets/PACS_class_vs_domain_nmi.png)

- GUIDE is implemented as an algorithm in the `domainbed/algorithms.py` file.
- See [Precompute Features](#precompute-features) for instructions on how to save the features (${\Psi}$) to disk.
- Follow DomainBed's instructions to run GUIDE on the datasets.

# Precompute Features
- Run `run_precompute.sh` by setting the layer and model arguments.
- The features are saved in the `domainbed/saved_feats` directory.

## Exmaple:
- For GUIDE-SD-2.1 on TerraIncognita dataset, run:

1. Get Stable Diffusion 2.1 Features at up_ft:1 and timestep 50 for TerraIncognita
```bash
bash run_precompute.sh
```

2. Run DomainBed on a single environment for the above setting
```bash
python3 -m domainbed.scripts.train_precompute\
       --data_dir=domainbed/data/\
       --algorithm GUIDE\
       --dataset TerraIncognita\
       --test_env 3\
       --hparams '{"model_name": "stabilityai/stable-diffusion-2-1-base", "feature_model": "diffusion", "timestep": 50, "num_clusters": 5}'
```

3. Create a sweep
```bash
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
```
This code is built on top of [Domainbed](https://github.com/facebookresearch/DomainBed/tree/main), visit Domainbed for more details on running training sweeps, hyperparameter configurations, etc.

## Citation

```
@misc{thomas2025whatslatentleveragingdiffusion,
      title={What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization}, 
      author={Xavier Thomas and Deepti Ghadiyaram},
      year={2025},
      eprint={2503.06698},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.06698}, 
}
```

## License

This source code is released under the MIT license, included [here](LICENSE).
