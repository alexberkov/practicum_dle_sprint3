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

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)
        self.mass_proj = nn.Linear(1, config.HIDDEN_DIM)

        self.attn = nn.MultiheadAttention(config.HIDDEN_DIM, num_heads=4, batch_first=True)

        self.regressor = nn.Sequential(
            nn.Linear(2 * config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_PROB),

            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_PROB),
            
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )

    def forward(self, input_ids, attention_mask, image, mass):
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:, 0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features).unsqueeze(1)
        image_emb = self.image_proj(image_features).unsqueeze(1)
        mass_emb = self.mass_proj(mass.unsqueeze(1)).unsqueeze(1)

        attn_emb_t, _ = self.attn(query=text_emb, key=image_emb, value=image_emb)
        attn_emb_m, _ = self.attn(query=mass_emb, key=image_emb, value=image_emb)

        fused_emb = torch.cat([attn_emb_t, attn_emb_m], dim=-1).squeeze(1)

        return self.regressor(fused_emb)
