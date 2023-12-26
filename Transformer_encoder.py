import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SharedWeightTransformer(nn.Module):
  transformer_encoder = nn.Sequential(
    nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model=512,
        dim_feedforward=1024,
        nhead=8,
      ),
      num_layers=1,
    ),
    nn.Linear(512, 256),
    nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model=256,
        dim_feedforward=1024,
        nhead=8,
      ),
      num_layers=1,
    ),
    nn.Linear(256, 128),
    nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model=128,
        dim_feedforward=1024,
        nhead=8,
      ),
      num_layers=1,
    ),
    nn.Linear(125, 256),
    nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model=256,
        dim_feedforward=1024,
        nhead=8,
      ),
      num_layers=1,
    ),
    nn.Linear(256, 384),
    nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model=384,
        dim_feedforward=1024,
        nhead=8,
      ),
      num_layers=1,
    ),
  )

  def __init__(self):
    super(SharedWeightTransformer, self).__init__()
    self.transformer_encoder_ = self.transformer_encoder

  def forward(self, src):
    return self.transformer_encoder_(src)
