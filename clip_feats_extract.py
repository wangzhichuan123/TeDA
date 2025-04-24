import argparse
from tqdm import tqdm
import time
import random
import numpy as np
import torch
from dataset.esb_core import ESBCoreDataset
from dataset.ntu_core import NTUCoreDataset
from dataset.mn40_core import MN40CoreDataset
from dataset.abo_core import ABOCoreDataset
import open_clip
import clip
import os


def setup_seed():
    seed = 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"random seed: {seed}")


def extract_feats(model, data_loader):
    """ """
    model.eval()
    feats = []
    labels = []
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        for batch in tqdm(data_loader):
            mv_imgs, category, _ = batch
            mv_imgs = mv_imgs.cuda()
            bz, n, c, h, w = mv_imgs.size()
            mv_imgs = mv_imgs.view(-1, c, h, w)
            mv_imgs = mv_imgs.half()

            mv_feat = model.encode_image(mv_imgs)
            mv_feat = mv_feat.view(bz, n, -1)

            feats.append(mv_feat.detach().cpu())
            labels.append(category.detach().cpu())

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


def main():

    # args
    parser = argparse.ArgumentParser()
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

    parser.add_argument("--open_clip", default=False)

    parser.add_argument("--n_view", default=24, type=int)

    args = parser.parse_args()

    print(args)

    setup_seed()

    if args.dataset == "esb":
        data_dir = "/mnt/sda/shared_datasets/3dosr_data/OS-ESB-core"
        query_dataset = ESBCoreDataset(
            data_dir, "query", modality="mv", n_view=args.n_view
        )
        target_dataset = ESBCoreDataset(
            data_dir, "target", modality="mv", n_view=args.n_view
        )

    elif args.dataset == "ntu":
        data_dir = "/mnt/sda/shared_datasets/3dosr_data/OS-NTU-core"
        query_dataset = NTUCoreDataset(data_dir, "query", modality="mv")
        target_dataset = NTUCoreDataset(data_dir, "target", modality="mv")

    elif args.dataset == "mn40":
        data_dir = "/mnt/sda/shared_datasets/3dosr_data/OS-MN40-core"
        query_dataset = MN40CoreDataset(
            data_dir, "query", modality="mv", n_view=args.n_view
        )
        target_dataset = MN40CoreDataset(
            data_dir, "target", modality="mv", n_view=args.n_view
        )

    elif args.dataset == "abo":
        data_dir = "/mnt/sda/shared_datasets/3dosr_data/OS-ABO-core"
        query_dataset = ABOCoreDataset(
            data_dir, "query", modality="mv", n_view=args.n_view
        )
        target_dataset = ABOCoreDataset(
            data_dir, "target", modality="mv", n_view=args.n_view
        )

    else:
        raise NotImplementedError
    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    target_loader = torch.utils.data.DataLoader(
        target_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    if args.open_clip:
        print(f"Using OpenCLIP {args.backbone} Model!")
        if args.backbone == "ViT-B/32":
            model_clip, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
        elif args.backbone == "ViT-L/14":
            model_clip, _, _ = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="laion2b_s32b_b82k"
            )
        else:
            raise NotImplementedError
        model_clip.cuda().eval()
    else:
        print(f"Using CLIP {args.backbone} Model!")
        model_clip, preprocess = clip.load(args.backbone)
        model_clip.eval()

    if True:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            # save features
            save_file_prefix = f"{args.dataset}"
            save_file_suffix = f"{args.backbone}".replace("/", "_")
            if args.open_clip:
                save_file_suffix += "_open_clip"
            if args.zero_shot:
                save_file_suffix += "_zs"
            if args.n_view != 24:
                save_file_suffix += f"_{args.n_view}"
            feats_query, labels_query = extract_feats(model_clip, query_loader)
            if not os.path.exists("image_feats"):
                os.makedirs("image_feats")
            np.save(
                f"image_feats/{save_file_prefix}_query_feats_{save_file_suffix}.npy",
                feats_query.numpy(),
            )
            np.save(
                f"image_feats/{save_file_prefix}_query_labels_{save_file_suffix}.npy",
                labels_query.numpy(),
            )
            feats_target, labels_target = extract_feats(model_clip, target_loader)
            np.save(
                f"image_feats/{save_file_prefix}_target_feats_{save_file_suffix}.npy",
                feats_target.numpy(),
            )
            np.save(
                f"image_feats/{save_file_prefix}_target_labels_{save_file_suffix}.npy",
                labels_target.numpy(),
            )
            print("Extract features done!")


if __name__ == "__main__":

    all_st = time.time()
    main()
    all_sec = time.time() - all_st
    print(
        f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes {all_sec%60:.2f}s!"
    )
    print("All done!")
