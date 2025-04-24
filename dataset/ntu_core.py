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


# the first 13 classes are seen classes, the rest are unseen classes
# seen will be used for training, query and target will be used for testing (all samples are unseen classes)
Categories2IDS = {
    "ball": 0,
    "balloon": 1,
    "book": 2,
    "cannon": 3,
    "cold_weapon___stick": 4,
    "frame": 5,
    "gun___pistol": 6,
    "headstone": 7,
    "plane___delta_wing": 8,
    "plant___leaf": 9,
    "plant___with_pot": 10,
    "table___square": 11,
    "watch": 12,
    "animal___dinosaur": 13,
    "animal___dog": 14,
    "animal___duck": 15,
    "animal___fish": 16,
    "animal___tetrapods": 17,
    "bed": 18,
    "bird": 19,
    "bottle": 20,
    "car___common": 21,
    "car___truck": 22,
    "chair___common": 23,
    "chair___sofa_multi-man": 24,
    "chair___swivel": 25,
    "chess": 26,
    "chip": 27,
    "clock": 28,
    "cold_weapon___knife": 29,
    "cold_weapon___long": 30,
    "cold_weapon___sword": 31,
    "computer": 32,
    "container": 33,
    "cup": 34,
    "cycle___bike": 35,
    "cycle___moto": 36,
    "door": 37,
    "gun___musket": 38,
    "gun___submachine": 39,
    "helmet": 40,
    "human_stand": 41,
    "hydrant": 42,
    "insect___fly": 43,
    "insect___polypod": 44,
    "lamp___floorlamp": 45,
    "lamp___tablelamp": 46,
    "missle": 47,
    "orchestral": 48,
    "pen": 49,
    "plane___backswept_wing": 50,
    "plane___forwardswept_wing": 51,
    "plane___giant": 52,
    "plane___helicopter": 53,
    "plane___straight_wing": 54,
    "plant___flower": 55,
    "plant___tree": 56,
    "plant___weed": 57,
    "ring": 58,
    "ship___galleon": 59,
    "ship___modern": 60,
    "table___round": 61,
    "tank": 62,
    "tool___hammer": 63,
    "tool___screwdriver": 64,
    "wheel": 65,
    "zeppelin": 66,
}


class NTUCoreDataset(Dataset):
    def __init__(self, data_dir, split, modality="mv"):
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

    def __fetch_img_list(self, instance_path, n_view=24):
        all_filenames = sorted(list(Path(instance_path).glob("image/h_*.jpg")))
        all_view = len(all_filenames)
        filenames = all_filenames[:: all_view // n_view][:n_view]
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
            img_list = self.__fetch_img_list(instance_path, 24)
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
            pc = self.__read_pointcloud(self.__fetch_pt_path(instance_path, 1024))
            if self.split == "train":
                np.random.shuffle(pc)
            pc = self.transform(pc)
            label = self.label2idx[label]
            return pc, label, instance_path


class PKSampler(data.sampler.Sampler):
    """
    PK sample according to person identity
    Arguments:
        data_source(lightreid.data.ReIDdataset)
        k(int): sample k images of each person
    """

    def __init__(self, data_source, k):
        # parameters
        self.data_source = data_source
        self.pid_idx = "label"
        self.k = k
        # multi-processing
        # self.mp = dist.is_available()
        self.mp = False
        if self.mp:
            self.rank = dist.get_rank()
            self.word_size = dist.get_world_size()
        # init
        self.samples = self.data_source.samples
        self.class_dict = self._tuple2dict(self.samples)

    def __iter__(self):
        # import pdb; pdb.set_trace()
        self.sample_list = self._generate_list(self.class_dict)
        if not self.mp:
            return iter(self.sample_list)
        else:
            start = self.rank
            return itertools.islice(self.sample_list, start, None, self.word_size)

    def __len__(self):
        return len(self.sample_list)

    def _tuple2dict(self, inputs):
        """
        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (imagespath_2, class_index_2), ...]
        :return: dict, {class_index_i: [samples_index1, samples_index2, ...]}
        """
        dict = {}
        # import pdb; pdb.set_trace()
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.pid_idx]
            if class_index not in list(dict.keys()):
                dict[class_index] = [index]
            else:
                dict[class_index].append(index)
        return dict

    def _generate_list(self, dict):
        sample_list = []

        dict_copy = dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)
        while len(sample_list) < len(self.samples):
            for key in keys:
                if len(sample_list) >= len(self.samples):
                    break
                value = dict_copy[key]
                if len(value) >= self.k:
                    random.shuffle(value)
                    sample_list.extend(value[0 : self.k])
                else:
                    value = value * self.k
                    random.shuffle(value)
                    sample_list.extend(value[0 : self.k])
        return sample_list


if __name__ == "__main__":
    data_dir = "../data/OS-NTU-core"
    train_dataset = NTUCoreDataset(data_dir, "train")
    seen_categories = train_dataset.label_list
    print(len(train_dataset))
    query_dataset = NTUCoreDataset(data_dir, "query")
    target_dataset = NTUCoreDataset(data_dir, "target")
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

    train_dataset = NTUCoreDataset(data_dir, "train", modality="point")
    pc, label = train_dataset[0]
    train_dataset = NTUCoreDataset(data_dir, "train", modality="vox")
    vox, label = train_dataset[0]