import re

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW


def set_requires_grad(model, unfreeze_layers: str):
    pattern = re.compile(unfreeze_layers)
    for name, param in model.named_parameters():
        if pattern.search(name):
            param.requires_grad = True
        else:
            param.requires_grad = False


def train(model, config, train_loader, val_loader, device):
    set_requires_grad(model.text_model, unfreeze_layers=config.TEXT_MODEL_UNFREEZE)
    set_requires_grad(model.image_model, unfreeze_layers=config.IMAGE_MODEL_UNFREEZE)

    optimizer = AdamW([
        {'params': model.text_model.parameters(), 'lr': config.TEXT_LR},
        {'params': model.image_model.parameters(), 'lr': config.IMAGE_LR},
        {'params': model.image_proj.parameters(), 'lr': config.FUSION_LR},
        {'params': model.text_proj.parameters(), 'lr': config.FUSION_LR},
        {'params': model.mass_proj.parameters(), 'lr': config.FUSION_LR},
        {'params': model.regressor.parameters(), 'lr': config.REGRESSOR_LR}
    ])

    criterion = nn.SmoothL1Loss()

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
            results = batch['result'].to(device)

            optimizer.zero_grad()
            predictions = model(**inputs)
            loss = criterion(predictions, results)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = validate(model, criterion, val_loader, device)

        print(f"Epoch {epoch + 1}/{config.EPOCHS} | "
              f"Loss (train): {train_loss:.5f} | "
              f"Loss (val): {val_loss :.5f}")

        if val_loss < config.TARGET_LOSS:
            print('Достигнуто необходимое качество.')
            torch.save(model.state_dict(), config.SAVE_PATH)
            break


def validate(model, criterion, val_loader, device, verbose=False):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device)
            }
            results = batch['result'].to(device)

            predictions = model(**inputs)
            loss = criterion(predictions, results)
            val_loss += loss.item()

            if verbose:
                print(loss)

    return val_loss / len(val_loader)
