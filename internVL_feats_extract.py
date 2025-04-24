import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
import clip
from tqdm import tqdm
from pathlib import Path
import argparse
from dataset.esb_core import ESBCoreDataset
from dataset.ntu_core import NTUCoreDataset
from dataset.mn40_core import MN40CoreDataset
from dataset.abo_core import ABOCoreDataset
import open_clip

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_osr_images(folder_path, n_view):
    images = []
    num_patches_list = []
    all_filenames = sorted(
        list(Path(folder_path).glob("image/h_*.jpg")),
        key=lambda x: int(x.stem.split("_")[1]),
    )
    all_view = len(all_filenames)
    # import pdb; pdb.set_trace()
    filenames = all_filenames[:: all_view // n_view][:n_view]
    for filename in filenames:
        # image_path = os.path.join(folder_path, filename)
        img = load_image(filename, max_num=12).to(torch.bfloat16).cuda()
        images.append(img)
        num_patches_list.append(img.size(0))

    pixel_values = torch.cat(images, dim=0)
    return pixel_values, num_patches_list


def load_osr_dataset(
    args, query_loader, target_loader, model, tokenizer, generation_config, clip_model
):

    save_file_suffix = f"{args.backbone}".replace("/", "_")
    save_file_suffix = save_file_suffix + "_Q" + f"{args.question}"
    if args.open_clip:
        save_file_suffix += "_open_clip"
    if args.zero_shot:
        save_file_suffix += "_zs"
    if args.n_view != 24:
        save_file_suffix += f"_{args.n_view}"

    if args.question == 1:
        question = "<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\nThere are images of an object from different angles.Describe this object in one sentence."
    elif args.question == 2:
        question = "<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\nThere are images of an object from different angles.Describe this object's shape information in one sentence."

    if args.open_clip:
        if args.backbone == "ViT-B/32":
            clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        elif args.backbone == "ViT-L/14":
            clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

    if not os.path.exists("text_feats"):
        os.makedirs("text_feats")

    text_features_list = []
    for _, _, path in tqdm(query_loader):
        for image_path in path:
            pixel_values, num_patches_list = load_osr_images(image_path, args.n_view)

            response, _ = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                if args.open_clip:
                    texts = clip_tokenizer(response).cuda()
                else:
                    texts = clip.tokenize(response).cuda()
                text_feature = clip_model.encode_text(texts)
                text_features_list.append(text_feature.detach().cpu())

    text_features_array = torch.cat(text_features_list, dim=0)

    np.save(
        f"text_feats/{args.dataset}_query_feats_{save_file_suffix}.npy",
        text_features_array,
    )

    print(f"{args.dataset} query text features done!")

    text_features_list = []
    for _, _, path in tqdm(target_loader):
        for image_path in path:
            pixel_values, num_patches_list = load_osr_images(image_path, args.n_view)

            response, _ = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                if args.open_clip:
                    texts = clip_tokenizer(response).cuda()
                else:
                    texts = clip.tokenize(response).cuda()
                text_feature = clip_model.encode_text(texts)
                text_features_list.append(text_feature.detach().cpu())

    text_features_array = torch.cat(text_features_list, dim=0)

    np.save(
        f"text_feats/{args.dataset}_target_feats_{save_file_suffix}.npy",
        text_features_array,
    )

    print(f"{args.dataset} target text features done!")


if __name__ == "__main__":

    path = "OpenGVLab/InternVL2-4B"
    model = (
        AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    parser = argparse.ArgumentParser()
    # abo -> mn40: abo as query, mn40 as target
    parser.add_argument(
        "--dataset",
        type=str,
        default="esb",
        help="esb, ntu, mn40, abo, abo-mn40, mn40-abo",
    )
    parser.add_argument("--backbone", default="ViT-B/32", type=str)

    parser.add_argument("--question", default=1, type=int)

    parser.add_argument(
        "--zero_shot",
        default=False,
    )

    parser.add_argument("--n_view", default=24, type=int)

    parser.add_argument("--open_clip", default=False)

    args = parser.parse_args()

    print(args)

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

    load_osr_dataset(
        args,
        query_loader,
        target_loader,
        model,
        tokenizer,
        generation_config,
        model_clip,
    )
