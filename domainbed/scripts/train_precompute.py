# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import (
    InfiniteDataLoader,
    FastDataLoader,
    FastDataLoader_no_shuffle,
)

from .helpers import *
from .clustering import Faiss_Clustering
from sklearn.kernel_ridge import KernelRidge
import random
from sklearn.metrics.pairwise import euclidean_distances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain generalization")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset", type=str, default="RotatedMNIST")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument(
        "--task",
        type=str,
        default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"],
    )
    parser.add_argument("--hparams", type=str, help="JSON-serialized hparams dict")
    parser.add_argument(
        "--hparams_seed",
        type=int,
        default=0,
        help='Seed for random hparams (0 means "default hparams")',
    )
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and " "random_hparams).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of steps. Default is dataset-dependent.",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="Checkpoint every N steps. Default is dataset-dependent.",
    )
    parser.add_argument("--test_envs", type=int, nargs="+", default=[0])
    parser.add_argument("--output_dir", type=str, default="train_output")
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument(
        "--uda_holdout_fraction",
        type=float,
        default=0,
        help="For domain adaptation, % of test to use unlabeled for training.",
    )
    parser.add_argument("--skip_model_save", action="store_true")
    parser.add_argument("--save_model_every_checkpoint", action="store_true")
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, "out.txt"))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, "err.txt"))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print("Args:")
    for k, v in sorted(vars(args).items()):
        print("\t{}: {}".format(k, v))

    cluster = False

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(
            args.algorithm,
            args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed),
        )
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print("HParams:")
    for k, v in sorted(hparams.items()):
        print("\t{}: {}".format(k, v))

    try:
        cleaned_model_name = hparams["model_name"].replace("/", "-")
    except:
        cleaned_model_name = "facebook-DiT-XL-2-512_dit_50"
        hparams["feature_model"] = "diffusion"
        hparams["timestep"] = "50"
    if args.dataset == "TerraIncognita":
        feat_save_name = f"domainbed/saved_feats/{cleaned_model_name}_{hparams['feature_model']}_{hparams['timestep']}/terra_incognita"
    elif args.dataset == "OfficeHome":
        feat_save_name = f"domainbed/saved_feats/{cleaned_model_name}_{hparams['feature_model']}_{hparams['timestep']}/office_home"
    elif args.dataset == "DomainNet":
        feat_save_name = f"domainbed/saved_feats/{cleaned_model_name}_{hparams['feature_model']}_{hparams['timestep']}/domain_net"
    else:
        feat_save_name = f"domainbed/saved_feats/{cleaned_model_name}_{hparams['feature_model']}_{hparams['timestep']}/{args.dataset}"
    hparams["feat_save_name"] = feat_save_name

    if "GUIDE" in args.algorithm:
        if hparams["feature_model"] == "dit":
            if hparams["cluster_space"] == "cluster":
                hparams["cluster_dim"] = 1152
            else:
                hparams["cluster_dim"] = 2048
            if "GUIDE" in args.algorithm and hparams["align_reproj"]:
                hparams["cluster_dim"] = 2048
        else:
            if hparams["cluster_space"] == "cluster":
                hparams["cluster_dim"] = 1280
            else:
                hparams["cluster_dim"] = 2048
            if "GUIDE" in args.algorithm and hparams["align_reproj"]:
                hparams["cluster_dim"] = 2048

        if hparams["random_centroids"]:
            args.random_centroids = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)
    else:
        raise NotImplementedError

    print(f"{args.dataset}_precomputefeat")
    dataset = vars(datasets)[f"{args.dataset}_precomputefeat"](
        args.data_dir, args.test_envs, hparams
    )
    # if args.dataset in vars(datasets):
    #     dataset = vars(datasets)[f"{args.dataset}_precomputefeat"](args.data_dir, args.test_envs, hparams)
    # else:
    #     raise NotImplementedError

    # if "GUIDE" in args.algorithm:
    #     num_clusters = (
    #         args.num_clusters or hparams["num_clusters"] * dataset.num_classes
    #     )  # Set number of clusters
    #     print("NUM CLUSTERS: ", num_clusters)

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    test_data_sep = []
    train_data_sep = []
    # train_data_sep_diff = []
    # test_data_sep_diff = []
    eval_loader_names = []
    in_splits = []
    out_splits = []
    train_domain_labels = []
    for env_i, (env) in enumerate(dataset):
        print(f"Processing env {env_i}")
        uda = []
        out, in_ = misc.split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
        )
        # out_diff, in_diff = misc.split_dataset(
        #     diff_env,
        #     int(len(diff_env) * args.holdout_fraction),
        #     misc.seed_hash(args.trial_seed, env_i),
        # )
        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(
                in_,
                int(len(in_) * args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i),
            )
            # uda_diff, in_diff = misc.split_dataset(
            #     in_diff,
            #     int(len(in_diff) * args.uda_holdout_fraction),
            #     misc.seed_hash(args.trial_seed, env_i),
            # )
        test_data_sep.append(in_)
        # test_data_sep_diff.append(in_diff)
        eval_loader_names += ["env{}_in".format(env_i)]
        test_data_sep.append(out)
        # test_data_sep_diff.append(out_diff)
        eval_loader_names += ["env{}_out".format(env_i)]
        if env_i not in args.test_envs:
            train_data_sep.append(in_)
            # train_data_sep_diff.append(in_diff)
            train_domain_labels.extend([env_i] * len(in_))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_data = MyDataloader(train_data_sep)  # Concat train data
    test_data = MyDataloader(test_data_sep)  # Concat test data

    # train_data_diff = MyDataloader(train_data_sep_diff)  # Concat train data
    # test_data_diff = MyDataloader(test_data_sep_diff)  # Concat test data

    print(len(train_data), len(test_data))

    # print image sizes
    img = train_data[0][0]
    print("Image size: ", img[0].size())

    # img, _ = train_data_diff[0][0]
    # print("Image size: ", img.size())

    len_train_data = len(train_data)
    len_test_data = len(test_data)

    # Loaders to perform clustering
    if "GUIDE" in args.algorithm:
        train_loader = FastDataLoader_no_shuffle(
            dataset=train_data,
            batch_size=128,
            num_workers=4,
        )
        test_loader = FastDataLoader_no_shuffle(
            dataset=test_data,
            batch_size=128,
            num_workers=4,
        )

    # DomainBed dataloaders
    train_idx_split = get_data_split_idx(train_data_sep)
    train_loaders = [
        InfiniteDataLoader(
            dataset=torch.utils.data.Subset(train_data, idx),
            weights=None,
            batch_size=hparams["batch_size"],
            num_workers=0,
        )
        for idx in train_idx_split
    ]
    train_minibatches_iterator = zip(*train_loaders)

    test_idx_split = get_data_split_idx(test_data_sep)
    eval_loaders = [
        FastDataLoader(
            dataset=torch.utils.data.Subset(test_data, idx),
            batch_size=64,
            num_workers=0,
        )
        for idx in test_idx_split
    ]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(args.test_envs),
        hparams,
    )

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    # uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = int(len(train_data) / hparams["batch_size"])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    from sklearn.cluster import KMeans

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict(),
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    from sklearn.cluster import KMeans
    from scipy.spatial.distance import (
        cdist,
    )  # Optional, if you want a vectorized assignment

    def return_centroids(
        algorithm,
        num_clusters,  # Number of clusters per environment
        step,
        timestep,
        cluster_labels_train=None,
        cluster_labels_test=None,
    ):
        print(f"Return Centroids, with num clusters: {num_clusters}, at step: {step}")

        # -------------------------------
        # Extract features and labels (Oracle version)
        # -------------------------------
        with torch.no_grad():
            org_train_features, train_features, train_labels = (
                get_features_precompute(
                    train_loader, algorithm, len_train_data, 128, device, timestep
                )
            )
            org_test_features, test_features, test_labels = (
                get_features_precompute(
                    test_loader, algorithm, len_test_data, 128, device, timestep
                )
            )
            train_labels = torch.tensor(train_labels, dtype=torch.long)
            test_labels = torch.tensor(test_labels, dtype=torch.long)

        if cluster_labels_train is None:
            # Cluster all training features together
            print(f"Clustering at step: {step}, with num clusters: {num_clusters}")
            kmeans = KMeans(
                n_clusters=num_clusters,
                init="k-means++",
                random_state=random.randint(0, 10000),
            )  # Initialize KMeans with random seed based on step.
            kmeans.fit(train_features)

            cluster_labels_train = kmeans.labels_
        else:
            print("Using previous cluster labels")
        print("Cluster labels train: ", cluster_labels_train)
        train_images_lists = get_images_list(
            num_clusters, len_train_data, cluster_labels_train
        )

        # Compute diffusion-space centroids for training
        train_cluster_centroids = np.zeros((len_train_data, train_features.shape[1]))
        for i, indx in enumerate(train_images_lists):
            if len(indx) > 0:
                train_cluster_centroids[indx] = train_features[indx].mean(axis=0)

        # Compute original-space centroids for training
        train_org_centroids = np.zeros((len_train_data, org_train_features.shape[1]))
        for i, indx in enumerate(train_images_lists):
            if len(indx) > 0:
                train_org_centroids[indx] = org_train_features[indx].mean(axis=0)

        if cluster_labels_test is None:
            cluster_labels_test = kmeans.predict(test_features)
        else:
            print("Using previous cluster labels for test")
        test_images_lists = get_images_list(
            num_clusters, len_test_data, cluster_labels_test
        )

        # Compute diffusion-space centroids for testing
        test_cluster_centroids = np.zeros((len_test_data, test_features.shape[1]))
        for i, indx in enumerate(test_images_lists):
            if len(indx) > 0:
                test_cluster_centroids[indx] = test_features[indx].mean(axis=0)

        # Compute original-space centroids for testing
        test_org_centroids = np.zeros((len_test_data, org_test_features.shape[1]))
        for i, indx in enumerate(test_images_lists):
            if len(indx) > 0:
                test_org_centroids[indx] = org_test_features[indx].mean(axis=0)

        train_org_clusters = []
        train_cluster_clusters = []
        for i, indx in enumerate(train_images_lists):
            if len(indx) > 0:
                c_org = org_train_features[indx].mean(
                    axis=0
                )  # original space cluster centroid
                c_cluster = train_features[indx].mean(
                    axis=0
                )  # diffusion space cluster centroid
                train_org_clusters.append(c_org)
                train_cluster_clusters.append(c_cluster)

        train_org_clusters = np.stack(
            train_org_clusters, axis=0
        )  # (num_clusters, D_org)
        train_cluster_clusters = np.stack(
            train_cluster_clusters, axis=0
        )  # (num_clusters, D_cluster)

        # Align centroids
        # -------------------------------
        # Fit RBF Kernel Ridge Regression for Mapping
        # -------------------------------
        print("Fitting an RBF mapping from learned cluster space to original space (raw features).")

        # Compute pairwise distances for gamma calculation
        pairwise_dists = euclidean_distances(train_cluster_clusters, train_cluster_clusters)
        median_dist = np.median(pairwise_dists)
        gamma = 1.0 / (2 * median_dist**2)
        print("Gamma: ", gamma)

        # Fit Kernel Ridge Regression
        reg = KernelRidge(kernel="rbf", alpha=1.0, gamma=gamma)
        reg.fit(train_cluster_clusters, train_org_clusters)

        # -------------------------------
        # Transform Training and Test Centroids
        # -------------------------------
        train_centroids_aligned = reg.predict(train_cluster_centroids)
        test_centroids_aligned = reg.predict(test_cluster_centroids)

        print("Train Centroids Aligned: ", train_centroids_aligned.shape)
        print("Test Centroids Aligned: ", test_centroids_aligned.shape)

        train_centroids_aligned = torch.from_numpy(train_centroids_aligned).float()
        test_centroids_aligned = torch.from_numpy(test_centroids_aligned).float()

        return (
            train_centroids_aligned,
            test_centroids_aligned,
            cluster_labels_train,
            cluster_labels_test,
        )

    last_results_keys = None
    cluster_labels_train = None
    cluster_labels_test = None
    num_clusters = None
    cluster = False
    if "GUIDE" in args.algorithm:
        cluster = True
    # log clustering step
    cluster_step = steps_per_epoch
    # cluster_step = [
    #     (x * cluster_step) for x in range(n_steps) if (x * cluster_step) <= n_steps
    # ]
    epochs = n_steps // steps_per_epoch
    cluster_step = [
        ((2**x) * steps_per_epoch) for x in range(epochs) if (2**x) <= epochs
    ]
    try:
        if hparams["const_num_clusters"]:
            num_clusters = hparams["num_clusters"]
        else:
            num_clusters = dataset.num_classes * hparams["num_clusters"]
    except:
        num_clusters = -1
    for step in range(start_step, n_steps):
        step_start_time = time.time()

        if cluster:
            if step == 0 or (step in cluster_step):
                (
                    train_centroids,
                    test_centroids,
                    cluster_labels_train,
                    cluster_labels_test,
                ) = return_centroids(
                    algorithm,
                    num_clusters,
                    step,
                    hparams["timestep"],
                    cluster_labels_train,
                    cluster_labels_test,
                )
        else:
            train_centroids = None
            test_centroids = None
        # minibatches_device = [
        #     (x.to(device), y.to(device)) for x, y in next(train_minibatches_iterator)
        # ]
        if cluster:
            minibatches_device = [
                (x[0].to(device), train_centroids[idx].to(device), x[2].to(device))
                for (x, idx) in next(train_minibatches_iterator)
            ]
        else:
            minibatches_device = [
                (x[0].to(device), x[2].to(device))
                for (x, idx) in next(train_minibatches_iterator)
            ]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device) for x, _ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                "step": step,
                "epoch": step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders)
            for name, loader in evals:
                acc = misc.accuracy(algorithm, loader, None, device, test_centroids)
                results[name + "_acc"] = acc

            results["mem_gb"] = torch.cuda.max_memory_allocated() / (
                1024.0 * 1024.0 * 1024.0
            )

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys], colwidth=12)

            results.update({"hparams": hparams, "args": vars(args)})

            epochs_path = os.path.join(args.output_dir, "results.jsonl")
            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f"model_step{step}.pkl")

    save_checkpoint("model.pkl")

    with open(os.path.join(args.output_dir, "done"), "w") as f:
        f.write("done")
