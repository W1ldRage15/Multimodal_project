import os
import random
import numpy as np
import timm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError
from functools import partial
from transformers import AutoModel, AutoTokenizer
from scripts.dataset import MultimodalDataset, collate_fn, get_transforms


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)
        self.mass_proj = nn.Linear(config.BATCH_SIZE, config.HIDDEN_DIM)

        self.fc = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(config.HIDDEN_DIM // 2, config.NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, image, mass):
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:,  0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)
        mass_emb = self.mass_proj(mass)

        fused_emb = text_emb * image_emb * mass_emb

        logits = self.fc(fused_emb)
        return logits


def train(config, device):
    seed_everything(config.SEED)
    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    set_requires_grad(model.text_model,
                      unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)
    optimizer = AdamW([{
        'params': model.text_model.parameters(),
        'lr': config.TEXT_LR
    }, {
        'params': model.image_model.parameters(),
        'lr': config.IMAGE_LR
    }, {
        'params': model.fc.parameters(),
        'lr': config.FC_LR
    }, {
        'params': model.mass_proj.parameters(),
        'lr': config.MASS_LR
    }
    ])
    criterion = nn.L1Loss()
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")
    train_dataset = MultimodalDataset(config, transforms)
    val_dataset = MultimodalDataset(config, val_transforms, ds_type="val")
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              drop_last = True,
                              collate_fn=partial(collate_fn,
                                                 tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            drop_last = True,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer))

    MAE_train = MeanAbsoluteError().to(device)
    MAE_val = MeanAbsoluteError().to(device)
    best_mae_val = 0.60
    print("Training started")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device)
            }
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(**inputs)
            loss = criterion(logits, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _ = MAE_train(logits, labels.unsqueeze(1))

        train_mae = MAE_train.compute().cpu().numpy()
        val_mae = validate(model, val_loader, device, MAE_val)
        MAE_val.reset()
        MAE_train.reset()

        if val_mae < best_mae_val:
            print(
                f"Epoch {epoch}/{config.EPOCHS-1} | avg_Loss: {total_loss/len(train_loader):.4f} | Train MAE: {train_mae :.4f}| Val MAE: {val_mae :.4f} - New best model"
            )
            best_mae_val = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)
        else:
            print(
                f"Epoch {epoch}/{config.EPOCHS-1} | avg_Loss: {total_loss/len(train_loader):.4f} | Train MAE: {train_mae :.4f}| Val MAE: {val_mae :.4f}"
            )


def validate(model, val_loader, device, MAE_val):
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device)
            }
            labels = batch['label'].to(device)

            logits = model(**inputs)
            _ = MAE_val(logits, labels.unsqueeze(1))

    return MAE_val.compute().cpu().numpy()


def test(config, device):
    print("Testing started")
    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    state_dict = torch.load(config.LOAD_PATH)
    model.load_state_dict(state_dict)
    test_transforms = get_transforms(config, ds_type="test")
    test_dataset = MultimodalDataset(config, test_transforms, ds_type="test")
    test_loader = DataLoader(test_dataset,
                        batch_size=config.BATCH_SIZE,
                        shuffle=False,
                        drop_last = True,
                        collate_fn=partial(collate_fn,
                                            tokenizer=tokenizer))
    MAE_test = MeanAbsoluteError().to(device)
    model.eval()
    
    with torch.no_grad():
        avg_MAE = 0.0 
        for batch in test_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device)
            }
            labels = batch['label'].to(device)
            logits = model(**inputs)

            _ = MAE_test(logits, labels.unsqueeze(1))
            test_MAE = MAE_test.compute().cpu().numpy()
            avg_MAE += test_MAE

        print(f"avg_MAE: {avg_MAE/len(test_loader):.4f}")
        print("Top 5 with MAX MAE:")
        mae_value = (torch.abs(logits - labels.unsqueeze(1)))
        for _ in range(5):
            id_max_mae = torch.argmax(mae_value)
            try:
                img = Image.open(f"data/images/{batch['dish_id'][id_max_mae]}/rgb.png").convert('RGB')
                plt.figure(figsize=(3,3))
                plt.imshow(img)
                plt.axis('off')
                plt.title(batch['dish_id'][id_max_mae])
            except:
                print(f"Id: {batch['dish_id'][id_max_mae]} - No image")

            mask = torch.ones_like(mae_value, dtype=torch.bool)
            mask[id_max_mae] = False
            mae_value = mae_value[mask]
