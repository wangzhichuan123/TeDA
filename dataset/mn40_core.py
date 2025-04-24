import os
import random
import itertools

import glob
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import open3d as o3d
from .pc_transforms import (
    PointsToTensor,
    PointCloudScaling,
    PointCloudCenterAndNormalize,
    PointCloudRotation,
)
import torch.utils.data as data
import torch.distributed as dist


# the first 8 classes are seen classes, the rest are unseen classes
# seen will be used for training, query and target will be used for testing (all samples are unseen classes)
Categories2IDS = {
    "airplane": 0,
    "flower_pot": 1,
    "glass_box": 2,
    "keyboard": 3,
    "monitor": 4,
    "night_stand": 5,
    "sink": 6,
    "table": 7,
    "bathtub": 8,
    "bed": 9,
    "bench": 10,
    "bookshelf": 11,
    "bottle": 12,
    "bowl": 13,
    "car": 14,
    "chair": 15,
    "cone": 16,
    "cup": 17,
    "curtain": 18,
    "desk": 19,
    "door": 20,
    "dresser": 21,
    "guitar": 22,
    "lamp": 23,
    "laptop": 24,
    "mantel": 25,
    "person": 26,
    "piano": 27,
    "plant": 28,
    "radio": 29,
    "range_hood": 30,
    "sofa": 31,
    "stairs": 32,
    "stool": 33,
    "tent": 34,
    "toilet": 35,
    "tv_stand": 36,
    "vase": 37,
    "wardrobe": 38,
    "xbox": 39,
}


class MN40CoreDataset(Dataset):
    def __init__(self, data_dir, split, modality="mv", n_view=24):
        super(MN40CoreDataset, self).__init__()
        assert split in ["train", "query", "target"]
        assert modality in ["mv", "vox", "point"]

        self.data_dir = data_dir
        self.split = split
        self.modal = modality
        self.samples, self.label_list = self.load_data()
        self.label2idx = {
            label: Categories2IDS[label] for _, label in enumerate(self.label_list)
        }
        self.idx2label = {
            Categories2IDS[label]: label for _, label in enumerate(self.label_list)
        }
        self.num_classes = len(self.label_list)
        self.n_view = n_view

        if split == "train":
            # transform
            if self.modal == "mv":
                self.img_size = 224
                self.transform = T.Compose(
                    [
                        T.RandomResizedCrop(self.img_size),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                    ]
                )

            elif self.modal == "point":
                # import pdb; pdb.set_trace()
                # transform pc to tensor
                self.transform = T.Compose(
                    [
                        PointsToTensor(),
                        PointCloudScaling(scale=[0.9, 1.1]),
                        PointCloudCenterAndNormalize(gravity_dim=1),
                        PointCloudRotation(angle=[0.0, 1.0, 0.0]),
                    ]
                )

        elif split == "query" or split == "target":
            # query and target: both has seen and unseen classes
            # import pdb; pdb.set_trace()
            if self.modal == "mv":
                self.img_size = 224
                self.transform = T.Compose(
                    [
                        T.Resize(self.img_size),
                        T.ToTensor(),
                    ]
                )
            elif self.modal == "point":
                self.transform = T.Compose(
                    [
                        PointsToTensor(),
                        PointCloudCenterAndNormalize(gravity_dim=1),
                    ]
                )
            else:
                print("voxel doest not need any transformation")
        else:
            raise NotImplementedError

    def __fetch_img_list(self, instance_path):
        all_filenames = sorted(
            list(Path(instance_path).glob("image/h_*.jpg")),
            key=lambda x: int(x.stem.split("_")[1]),
        )
        all_view = len(all_filenames)
        # import pdb; pdb.set_trace()
        filenames = all_filenames[:: all_view // self.n_view][: self.n_view]
        return filenames

    def __fetch_pt_path(self, instance_path, n_pt):
        return Path(instance_path) / "pointcloud" / f"pt_{n_pt}.pts"

    def __fetch_vox_path(self, instance_path, d_vox):
        return Path(instance_path) / "voxel" / f"vox_{d_vox}.ply"

    def __read_vox(self, vox_path, d_vox):
        vox_3d = o3d.io.read_voxel_grid(str(vox_path))
        vox_idx = torch.from_numpy(
            np.array([v.grid_index - 1 for v in vox_3d.get_voxels()])
        ).long()
        vox = torch.zeros((d_vox, d_vox, d_vox))
        vox[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]] = 1
        return vox.unsqueeze(0)

    def __read_images(self, path_list):
        imgs = [Image.open(v).convert("RGB") for v in path_list]
        return imgs

    def __read_pointcloud(self, pc_path):
        pt = np.asarray(o3d.io.read_point_cloud(str(pc_path)).points)
        pt = pt - np.expand_dims(np.mean(pt, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(pt**2, axis=1)), 0)
        pt = pt / dist
        return pt

    def load_data(self):
        if self.split == "train":
            data_root = Path(self.data_dir)
            train_list, seen_label = [], []
            for label_root in data_root.glob("train/*"):
                label_name = label_root.name
                for obj_path in label_root.glob("*/"):
                    train_list.append({"path": str(obj_path), "label": label_name})
                seen_label.append(label_name)
            seen_label = sorted(set(seen_label))
            return train_list, seen_label

        elif self.split == "query" or self.split == "target":
            split_file_path = Path(self.data_dir) / f"{self.split}_label.txt"

            label_list = []
            sample_list = []
            with open(split_file_path, "r") as fp:
                for line in fp.readlines():
                    obj_name, label_name = line.strip().split(",")
                    sample_list.append(
                        {
                            "path": str(Path(self.data_dir) / self.split / obj_name),
                            "label": label_name,
                        }
                    )
                    label_list.append(label_name)
            return sample_list, sorted(set(label_list))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        instance_path, label = sample["path"], sample["label"]
        # import pdb; pdb.set_trace()
        if self.modal == "mv":
            img_list = self.__fetch_img_list(instance_path)
            imgs = self.__read_images(img_list)
            imgs = [self.transform(img) for img in imgs]
            imgs = torch.stack(imgs)

            label = self.label2idx[label]

            return imgs, label, instance_path
        elif self.modal == "vox":
            vox = self.__read_vox(self.__fetch_vox_path(instance_path, 32), 32)
            label = self.label2idx[label]
            return vox, label, instance_path
        elif self.modal == "point":
            # import pdb; pdb.set_trace()
            # 1024 x 3
            pc = self.__read_pointcloud(self.__fetch_pt_path(instance_path, 1024))
            if self.split == "train":
                np.random.shuffle(pc)
            pc = self.transform(pc)
            label = self.label2idx[label]
            return pc, label, instance_path


if __name__ == "__main__":
    data_dir = "../data/OS-MN40-core"
    train_dataset = MN40CoreDataset(data_dir, "train")
    seen_categories = train_dataset.label_list
    print(len(train_dataset))
    query_dataset = MN40CoreDataset(data_dir, "query")
    target_dataset = MN40CoreDataset(data_dir, "target")
    seen_categories = train_dataset.label_list
    unseen_categories_query = query_dataset.label_list
    unseen_categories_target = target_dataset.label_list
    assert unseen_categories_query == unseen_categories_target
    all_categories = seen_categories + unseen_categories_target
    label2idx = {cat: idx for idx, cat in enumerate(all_categories)}
    print("now rewrite Categories2IDS above the line 100")
    import pdb

    pdb.set_trace()
    for data, label in train_dataset:
        print(data.shape, label)

    train_dataset = MN40CoreDataset(data_dir, "train", modality="point")
    pc, label = train_dataset[0]
    train_dataset = MN40CoreDataset(data_dir, "train", modality="vox")
    vox, label = train_dataset[0]
    # comment all the below code
    # for cat in query_dataset.label_list:
    #     if cat not in seen_categories:
    #         seen_categories.append(cat)
    # for cat in target_dataset.label_list:
    #     if cat not in seen_categories:
    #         seen_categories.append(cat)
    # import pdb; pdb.set_trace()
    # categories2idx = {cat: idx for idx, cat in enumerate(seen_categories)}
