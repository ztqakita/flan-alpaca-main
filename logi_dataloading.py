import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader

from logitorch.data_collators.ruletaker_collator import RuleTakerCollator
from logitorch.datasets.qa.ruletaker_dataset import RuleTakerDataset
from logitorch.pl_models.ruletaker import PLRuleTaker

train_dataset = RuleTakerDataset("depth-5", "train")
val_dataset = RuleTakerDataset("depth-5", "val")

ruletaker_collate_fn = RuleTakerCollator()

train_dataloader = DataLoader(
    train_dataset, batch_size=32, collate_fn=ruletaker_collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=32, collate_fn=ruletaker_collate_fn
)
