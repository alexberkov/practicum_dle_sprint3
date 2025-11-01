import numpy as np
import pandas as pd

from transformers import AutoTokenizer

from PIL import Image
import albumentations as A

import torch
import timm
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):

    def __init__(self, config, transforms, ds_type="train"):
        if ds_type == "train":
            self.df = pd.read_csv(config.TRAIN_DF_PATH)
        else:
            self.df = pd.read_csv(config.VAL_DF_PATH)
        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "ingredient_names"]
        result = self.df.loc[idx, "total_calories"]
        image = Image.open(self.df.loc[idx, "image_path"]).convert('RGB')
        mass = self.df.loc[idx, "total_mass"]
        dish_id = self.df.loc[idx, "dish_id"]

        transformed_image = self.transforms(image=np.array(image))["image"]

        return {"result": result, "image": transformed_image, "text": text, "mass": mass, "dish_id": dish_id}


def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    results = torch.tensor([item["result"] for item in batch], dtype=torch.float32).unsqueeze(1)
    masses = torch.tensor([item["mass"] for item in batch], dtype=torch.float32)

    tokenized_input = tokenizer(
        texts, return_tensors="pt", padding="max_length", truncation=True
    )

    dish_ids = [item["dish_id"] for item in batch]

    return {
        "result": results,
        "image": images,
        "mass": masses,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
        "dish_id": dish_ids
    }


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
    
    if ds_type == "train":
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.Affine(scale=(0.8, 1.2),
                         rotate=(-15, 15),
                         translate_percent=(-0.1, 0.1),
                         shear=(-10, 10),
                         fill=0,
                         p=0.8),
                A.CoarseDropout(num_holes_range=(2, 8),
                                hole_height_range=(int(0.07 * cfg.input_size[1]),
                                                   int(0.15 * cfg.input_size[1])),
                                hole_width_range=(int(0.1 * cfg.input_size[2]),
                                                  int(0.15 * cfg.input_size[2])),
                                fill=0,
                                p=0.5),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ]
        )

    return transforms
