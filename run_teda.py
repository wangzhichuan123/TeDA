import sys
import argparse
from tqdm import tqdm
import time
import random
import numpy as np
import scipy.spatial
import torch
import clip
from misc_utils.metric_tools import map_score, eval_all_metric, map_score_sim
from dataset.feat_dataset import FeatDatasetTrain, FeatDataset
import torch.nn as nn
import torch.nn.functional as F
import operator
import os
import yaml
import math
import shutil


def setup_seed():
    seed = 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print(f"random seed: {seed}")


@torch.no_grad()
def test_model_clip_osr3d_feats(
    args,
    query_loader,
    target_loader,
    eval_all=False,
):

    query_feats = []
    query_labels = []
    for mv_feats, category in tqdm(query_loader):
        mv_feats = mv_feats.cuda()

        mv_feats_mean = mv_feats.mean(dim=1)  # 16 512
        feats = mv_feats_mean

        query_feats.append(feats.detach().cpu())
        query_labels.append(category.detach().cpu())

    query_feats = torch.cat(query_feats, dim=0).cuda()
    query_labels = torch.cat(query_labels, dim=0).cuda()

    # target
    target_feats = []
    target_labels = []
    for mv_feats, category in tqdm(target_loader):
        mv_feats = mv_feats.cuda()
        mv_feats_mean = mv_feats.mean(dim=1)
        feats = mv_feats_mean
        target_feats.append(feats.detach().cpu())
        target_labels.append(category.detach().cpu())

    target_feats = torch.cat(target_feats, dim=0).cuda()
    target_labels = torch.cat(target_labels, dim=0).cuda()
    save_file_suffix = f"{args.backbone}".replace("/", "_")
    save_file_suffix = save_file_suffix + "_Q" + f"{args.question}"
    if args.zero_shot:
        save_file_suffix += "_zs"
    if args.open_clip:
        save_file_suffix += "_open_clip"
    if args.n_view != 24:
        save_file_suffix += f"_{args.n_view}"
    query_text = np.load(
        f"text_feats/{args.dataset}_query_feats_{save_file_suffix}.npy"
    )
    target_text = np.load(
        f"text_feats/{args.dataset}_target_feats_{save_file_suffix}.npy"
    )
    query_text = torch.tensor(query_text)
    query_text = query_text.cuda()
    target_text = torch.tensor(target_text)
    target_text = target_text.cuda()

    fusion_rate = args.fusion_rate
    combined_feats_query = query_feats + query_text * fusion_rate
    combined_feats_target = target_feats + target_text * fusion_rate
    tanh = nn.Tanh()

    combined_feats_query = tanh(combined_feats_query)
    combined_feats_target = tanh(combined_feats_target)

    retrieval_eval(
        args,
        combined_feats_query,
        combined_feats_target,
        query_labels,
        target_labels,
        eval_all,
    )


def image_opt(feat, init_classifier, plabel, lr=10, iter=2000, tau_i=0.04, alpha=0.6):
    ins, dim = feat.shape
    val, idx = torch.max(plabel, dim=1)
    mask = val > alpha
    plabel[mask, :] = 0
    plabel[mask, idx[mask]] = 1
    base = feat.T @ plabel
    classifier = init_classifier.clone()
    pre_norm = float("inf")
    for i in range(0, iter):
        prob = F.softmax(feat @ classifier / tau_i, dim=1)
        grad = feat.T @ prob - base
        temp = torch.norm(grad)
        if temp > pre_norm:
            lr /= 2.0
        pre_norm = temp
        classifier -= (lr / (ins * tau_i)) * grad
        classifier = F.normalize(classifier, dim=0)
    val, idx = torch.max(feat @ classifier, dim=1)
    return classifier


@torch.no_grad()
def retrieval_eval(args, query, target, query_lbls, target_lbls, eval_all=False):
    query_fts = query.squeeze()
    target_fts = target.squeeze()
    query_lbls = query_lbls
    target_lbls = target_lbls
    # dist_mat = scipy.spatial.distance.cdist(query_fts, target_fts, "cosine")

    query_fts = query_fts / query_fts.norm(dim=-1, keepdim=True)
    target_fts = target_fts / target_fts.norm(dim=-1, keepdim=True)
    query_fts = query_fts.T
    logits = target_fts @ query_fts

    tau_t = args.tau_t
    tau_i = args.tau_i
    alpha = 0.6

    plabel = F.softmax(logits / tau_t, dim=1)
    pc_classifier = image_opt(
        target_fts,
        query_fts,
        plabel,
        10,
        2000,
        tau_i,
        alpha,
    )

    dist_mat = target_fts @ pc_classifier
    # dist_mat = target_fts @ query_fts

    map_s = map_score_sim(dist_mat.T, query_lbls, target_lbls)

    # evaluate all metrics
    if eval_all:
        print("evaluate all metrics:")
        eval_all_metric(pc_classifier.T, target_fts, query_lbls, target_lbls)
    return map_s, map_s


def main():
    # args
    parser = argparse.ArgumentParser()
    # abo -> mn40: abo as query, mn40 as target
    parser.add_argument(
        "--dataset",
        type=str,
        default="esb",
        help="esb, ntu, mn40, abo, abo-mn40, mn40-abo",
    )
    parser.add_argument("--backbone", default="ViT-B/32", type=str)
    parser.add_argument(
        "--zero_shot",
        default=False,
    )
    parser.add_argument("--question", default=1, type=int)
    parser.add_argument("--open_clip", default=False)
    parser.add_argument("--n_view", default=24, type=int)
    args = parser.parse_args()

    # 加载 config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 获取当前数据集的参数
    dataset_cfg = config.get(args.dataset)
    if dataset_cfg is None:
        raise ValueError(f"Config for dataset '{args.dataset}' not found!")

    # 添加参数到 args 中
    args.fusion_rate = dataset_cfg["fusion_rate"]
    args.tau_t = dataset_cfg["tau_t"]
    args.tau_i = dataset_cfg["tau_i"]

    print(f"Loaded config for {args.dataset}: {dataset_cfg}")

    print(args)

    setup_seed()

    if True:
        save_file_suffix = f"{args.backbone}".replace("/", "_")
        if args.open_clip:
            save_file_suffix += "_open_clip"
        if args.zero_shot:
            save_file_suffix += "_zs"
        if args.n_view != 24:
            save_file_suffix += f"_{args.n_view}"

        # load features
        feats_query = np.load(
            f"image_feats/{args.dataset}_query_feats_{save_file_suffix}.npy"
        )
        labels_query = np.load(
            f"image_feats/{args.dataset}_query_labels_{save_file_suffix}.npy"
        )
        feats_target = np.load(
            f"image_feats/{args.dataset}_target_feats_{save_file_suffix}.npy"
        )
        labels_target = np.load(
            f"image_feats/{args.dataset}_target_labels_{save_file_suffix}.npy"
        )
        print("Load features done!")

        query_dataset = FeatDataset(feats_query, labels_query)
        target_dataset = FeatDataset(feats_target, labels_target)
        query_loader = torch.utils.data.DataLoader(
            query_dataset, batch_size=16, shuffle=False, num_workers=0
        )
        target_loader = torch.utils.data.DataLoader(
            target_dataset, batch_size=16, shuffle=False, num_workers=0
        )

        test_model_clip_osr3d_feats(
            args,
            query_loader,
            target_loader,
            eval_all=True,
        )

        print("Training done!")


if __name__ == "__main__":

    all_st = time.time()
    main()
    all_sec = time.time() - all_st
    print(
        f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes {all_sec%60:.2f}s!"
    )
    print("All done!")
