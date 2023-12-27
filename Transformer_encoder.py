import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

T5_SMALL_DIM = 512
T5_LARGE_DIM = 1024


class SharedWeightTransformer(nn.Module):
  MODEL_DIM = T5_LARGE_DIM

  transformer_encoder = nn.Sequential(
    nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model=MODEL_DIM,
        dim_feedforward=MODEL_DIM * 2,
        nhead=1,
      ),
      num_layers=1,
    ),
    nn.Linear(MODEL_DIM, MODEL_DIM // 2),
    nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model=MODEL_DIM // 2,
        dim_feedforward=MODEL_DIM,
        nhead=1,
      ),
      num_layers=1,
    ),
    nn.Linear(MODEL_DIM // 2, MODEL_DIM // 4),
    nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model=MODEL_DIM // 4,
        dim_feedforward=MODEL_DIM // 2,
        nhead=1,
      ),
      num_layers=1,
    ),
    nn.Linear(MODEL_DIM // 4, MODEL_DIM // 2),
    nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model=MODEL_DIM // 2,
        dim_feedforward=MODEL_DIM,
        nhead=1,
      ),
      num_layers=1,
    ),
    nn.Linear(MODEL_DIM // 2, MODEL_DIM),
    nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model=MODEL_DIM,
        dim_feedforward=MODEL_DIM * 2,
        nhead=1,
      ),
      num_layers=1,
    ),
  )

  def __init__(self):
    super(SharedWeightTransformer, self).__init__()
    self.transformer_encoder_ = self.transformer_encoder

  def forward(self, src):
    return self.transformer_encoder_(src)
