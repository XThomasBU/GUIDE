import torch
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from tqdm import tqdm


class MyDataloader(torch.utils.data.Dataset):
    """
    Combine Seperated Datasets
    """

    def __init__(self, data_list):
        self.data = torch.utils.data.ConcatDataset(data_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor):
    # image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    image_aspect_ratio = "pad"
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean)
            )
            image = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def get_features(data_loader, model, N, batch_size, device):
    """
    Input: dataloader, model used (eg: Resnet50), N: Size of dataset
    Returns: features, labels, idx
    """
    model.train(False)
    for i, ((data, labels), idx) in enumerate(data_loader):
        features = (
            model.featurizer(data.to(device)).detach().cpu().numpy()
        )  # get features
        features = np.asarray(features.reshape(features.shape[0], -1), dtype=np.float32)
        if i == 0:
            features_ = np.zeros((N, features.shape[1]), dtype=np.float32)
            labels_ = np.zeros(N, dtype=np.float32)
        if i < N - 1:
            features_[i * batch_size : (i + 1) * batch_size] = features
            labels_[i * batch_size : (i + 1) * batch_size] = labels
        else:
            features_[i * batch_size :] = features  # last batch
            labels_[i * batch_size :] = labels
    return features_, labels_


def get_features_V2(data_loader, model, N, batch_size, device, timestep):
    """
    Input: dataloader, model used (eg: Resnet50), N: Size of dataset
    Returns: features, labels, idx
    """
    model.train(False)
    model.cluster_featurizer.feature_model.model = model.cluster_featurizer.feature_model.model.to(device)
    preprocess = model.cluster_featurizer.preprocess
    # for i, ((data, labels), idx) in enumerate(data_loader):
    # for i, ((data), idx) in tqdm(enumerate(data_loader), total=len(data_loader)):
    for i, ((data), idx) in enumerate(data_loader):
        data1, data2, labels = data[0], data[1], data[2]
        # org_features = model.featurizer(data1.to(device), num_layers=1).detach().cpu().numpy()  # get features
        org_features = model.featurizer(data1.to(device)).detach().cpu().numpy()  # get features


        # features = model.cluster_featurizer(data.to(device)).detach().cpu().numpy()  # get features
        # preprocess_data = process_images(data, preprocess)
        # print(preprocess_data.shape)
        features = model.cluster_featurizer.get_raw_features(data2.to(device), time_step=timestep).detach().cpu().numpy()  # get features
        features = np.asarray(features.reshape(features.shape[0], -1), dtype=np.float32)
        org_features = np.asarray(org_features.reshape(org_features.shape[0], -1), dtype=np.float32)
        if i == 0:
            features_ = np.zeros((N, features.shape[1]), dtype=np.float32)
            labels_ = np.zeros(N, dtype=np.float32)
            org_features_ = np.zeros((N, org_features.shape[1]), dtype=np.float32)
        if i < N - 1:
            features_[i * batch_size : (i + 1) * batch_size] = features
            labels_[i * batch_size : (i + 1) * batch_size] = labels
            org_features_[i * batch_size : (i + 1) * batch_size] = org_features
        else:
            features_[i * batch_size :] = features  # last batch
            labels_[i * batch_size :] = labels
            org_features_[i * batch_size :] = org_features
    return org_features_, features_, labels_

def get_features_precompute(data_loader, model, N, batch_size, device, timestep):
    """
    Input: dataloader, model used (eg: Resnet50), N: Size of dataset
    Returns: features, labels, idx
    """
    model.train(False)
    model.cluster_featurizer.feature_model.model = model.cluster_featurizer.feature_model.model.to(device)
    preprocess = model.cluster_featurizer.preprocess
    # for i, ((data, labels), idx) in enumerate(data_loader):
    # for i, ((data), idx) in tqdm(enumerate(data_loader), total=len(data_loader)):
    for i, ((data), idx) in enumerate(data_loader):
        data1, _, labels, path, feat_path, features = data[0], data[1], data[2], data[3], data[4], data[5]
        # org_features = model.featurizer(data1.to(device), num_layers=1).detach().cpu().numpy()  # get features
        org_features = model.featurizer(data1.to(device)).detach().cpu().numpy()  # get features


        # features = model.cluster_featurizer(data.to(device)).detach().cpu().numpy()  # get features
        # preprocess_data = process_images(data, preprocess)
        # print(preprocess_data.shape)
        # features = model.cluster_featurizer.get_raw_features(data2.to(device), time_step=timestep).detach().cpu().numpy()  # get features
        features = np.asarray(features.reshape(features.shape[0], -1), dtype=np.float32)
        org_features = np.asarray(org_features.reshape(org_features.shape[0], -1), dtype=np.float32)
        if i == 0:
            features_ = np.zeros((N, features.shape[1]), dtype=np.float32)
            labels_ = np.zeros(N, dtype=np.float32)
            org_features_ = np.zeros((N, org_features.shape[1]), dtype=np.float32)
        if i < N - 1:
            features_[i * batch_size : (i + 1) * batch_size] = features
            labels_[i * batch_size : (i + 1) * batch_size] = labels
            org_features_[i * batch_size : (i + 1) * batch_size] = org_features
        else:
            features_[i * batch_size :] = features  # last batch
            labels_[i * batch_size :] = labels
            org_features_[i * batch_size :] = org_features
    return org_features_, features_, labels_



def get_cluster_labels(clustering, features):
    _, I = clustering.kmeans.index.search(features.copy(order="C"), 1)
    cluster_labels = [int(n[0]) for n in I]
    return cluster_labels


def get_images_list(num_clusters, len_data, cluster_labels):
    images_lists = [[] for i in range(num_clusters)]
    for i in range(len_data):
        images_lists[cluster_labels[i]].append(i)
    return images_lists


def get_hparam(hparams, hparams_seed):
    """
    Function to Set hparam values
    """
    hparam_num = 0
    pca_offset = [0]
    num_clusters = [5]
    clust_epoch = [0]
    for p in pca_offset:
        for n in num_clusters:
            for e in clust_epoch:
                if hparams_seed == hparam_num:
                    hparams["num_clusters"] = n
                    hparams["offset"] = p
                    hparams["clust_epoch"] = e
                hparam_num += 1
    return hparams


def get_data_split_idx(dataset):
    l = 0
    idx_list = []
    for i in dataset:
        idx_list.append(list(range(l, len(i) + l)))
        l = len(i) + l
    return idx_list
