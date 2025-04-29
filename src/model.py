import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=9, model_dim=64, num_heads=4, num_layers=2, num_classes=2):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.input_proj(x)

        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer_encoder(x)
        x = x[:, 0, :]  # take CLS token

        out = self.fc(x)
        return out