import timm
import torch
import torch.nn as nn
from transformers import AutoModel


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.TEXT_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.IMAGE_DIM)
        self.mass_proj = nn.Linear(1, config.MASS_DIM)

        self.fusion = nn.Linear(
            config.TEXT_DIM + config.IMAGE_DIM + config.MASS_DIM, config.HIDDEN_DIM
        )

        self.regressor = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_PROB),

            nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM // 4),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_PROB),

            nn.Linear(config.HIDDEN_DIM // 4, 1)
        )

    def forward(self, input_ids, attention_mask, image, mass):
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:, 0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)
        mass_emb = self.mass_proj(mass.unsqueeze(1))

        fused_emb = self.fusion(torch.cat([text_emb, image_emb, mass_emb], dim=-1))

        return self.regressor(fused_emb)
