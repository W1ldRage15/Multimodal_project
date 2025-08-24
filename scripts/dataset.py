import torch
import albumentations as A
import timm
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer


class MultimodalDataset(Dataset):

    def __init__(self, config, transforms, ds_type="train"):
        if ds_type == "train":
            self.df = pd.read_csv(config.TRAIN_DF_PATH)
        else:
            self.df = pd.read_csv(config.TEST_DF_PATH)
        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dish_id = self.df.loc[idx, "dish_id"]
        text = self.df.loc[idx, "ing_names"]
        label = self.df.loc[idx, "total_calories_scaled"]
        mass = self.df.loc[idx, "total_mass_scaled"]
        img_path = self.df.loc[idx, "dish_id"]
        try:
            image = Image.open(f"data/images/{img_path}.jpg").convert('RGB')
        except:
            image = torch.randint(0, 255, (*self.image_cfg.input_size[1:],
                                           self.image_cfg.input_size[0])).to(
                                               torch.float32)

        image = self.transforms(image=np.array(image))["image"]
        return {"label": label, "image": image, "text": text, "mass": mass, "dish_id": dish_id,}


def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]
    dish_id = [item["dish_id"] for item in batch]
    mass = torch.FloatTensor([item["mass"] for item in batch])
    images = torch.stack([item["image"] for item in batch])
    labels = torch.FloatTensor([item["label"] for item in batch])

    tokenized_input = tokenizer(texts,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True)

    return {
        "label": labels,
        "image": images,
        "mass": mass,
        "dish_id": dish_id,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"]
    }


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.RandomCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Affine(scale=(0.8, 1.2),
                         rotate=(-15, 15),
                         translate_percent=(-0.1, 0.1),
                         shear=(-10, 10),
                         fill=0,
                         p=0.8),
                A.CoarseDropout(
                    num_holes_range=(2, 8),
                    hole_height_range=(int(0.07 * cfg.input_size[1]),
                                       int(0.15 * cfg.input_size[1])),
                    hole_width_range=(int(0.1 * cfg.input_size[2]),
                                      int(0.15 * cfg.input_size[2])),
                    fill=0,
                    p=0.5),
                A.ColorJitter(brightness=0.2,
                              contrast=0.2,
                              saturation=0.2,
                              hue=0.1,
                              p=0.7),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=config.SEED,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=config.SEED,
        )

    return transforms
