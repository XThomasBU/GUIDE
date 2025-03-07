import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from domainbed import datasets, hparams_registry, algorithms
from domainbed.lib.fast_data_loader import FastDataLoader_no_shuffle

class MyDataloader(torch.utils.data.Dataset):
    """
    Combine Separated Datasets
    """
    def __init__(self, data_list):
        self.data = torch.utils.data.ConcatDataset(data_list)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], idx

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="DomainNet", help="Dataset name")
parser.add_argument("--model_name", type=str, required=True, help="Model name, e.g., facebook/dinov2-large")
parser.add_argument("--feature_model", type=str, required=True, help="Feature model name, e.g., dino")
parser.add_argument("--timestep", type=int, default=50, help="Timestep for feature extraction")
args = parser.parse_args()

folder_dict={'PACS': 'PACS', "VLCS": "VLCS", "OfficeHome": "office_home", "TerraIncognita": "terra_incognita", "DomainNet": "domain_net"}

# Set hyperparameters
hparams = hparams_registry.default_hparams("GUIDE", args.dataset_name)
hparams["model_name"] = args.model_name
hparams["feature_model"] = args.feature_model
hparams["timestep"] = args.timestep

dataset = vars(datasets)[f"{args.dataset_name}_precomputefeat"]("domainbed/data", [-1], hparams)

# Prepare feature save path
cleaned_model_name = args.model_name.replace("/", "-")
feat_save_name = f"domainbed/saved_feats/{cleaned_model_name}_{args.feature_model}_{args.timestep}/{folder_dict[args.dataset_name]}"
hparams["feat_save_name"] = feat_save_name

if not os.path.exists(feat_save_name):
    os.makedirs(feat_save_name)

print(f"Feature save path: {feat_save_name}")
print(dataset)

hparams["cluster_dim"] = 2048
algorithm_class = algorithms.get_algorithm_class("GUIDE")
algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset), hparams)

data_sep = []
for env_i, env in enumerate(dataset):
    print(env_i)
    data_sep.append(env)

data = MyDataloader(data_sep)
data_loader = FastDataLoader_no_shuffle(data, batch_size=32, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
algorithm.train(False)
algorithm.cluster_featurizer.feature_model.model = algorithm.cluster_featurizer.feature_model.model.to(device)
preprocess = algorithm.cluster_featurizer.preprocess

for i, (x, y) in tqdm(enumerate(data_loader), total=len(data_loader)):
    img1, img2, class_labels, path, feat_path, _ = x
    features = algorithm.cluster_featurizer.get_raw_features(img2.to(device), time_step=hparams["timestep"]).detach().cpu().numpy()
    
    for idx, path_i in enumerate(path):
        save_path = path_i.split(f"{folder_dict[args.dataset_name]}/")[1].rsplit(".", 1)[0]
        feat_save_path = f"{feat_save_name}/{save_path}.npy"
        feate_save_dir = os.path.dirname(feat_save_path)
        if not os.path.exists(feate_save_dir):
            os.makedirs(feate_save_dir)
        with open(feat_save_path, "wb") as f:
            np.save(f, features[idx])
