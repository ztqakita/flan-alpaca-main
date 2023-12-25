from torch.utils.data.dataloader import DataLoader

from logitorch.data_collators.proofwriter_collator import ProofWriterQACollator, ProofWriterProofGenerationAllCollator
from logitorch.datasets.proof_qa.proofwriter_dataset import ProofWriterDataset

# ==================== ProofWriter MetaInfo====================

# PROOFWRITER_SUB_DATASETS = [
#     "birds-electricity",
#     "depth-0",
#     "depth-1",
#     "depth-2",
#     "depth-3",
#     "depth-3ext",
#     "depth-3ext-NatLang",
#     "depth-5",
#     "NatLang",
# ]
# PROOFWRITER_TASKS = [
#     "proof_generation_all",
#     "proof_generation_iter",
#     "implication_enumeration",
#     "abduction",
# ]

# abducution task need open_world_assumption=False

# =============================================================

train_dataset = ProofWriterDataset(dataset_name="depth-5", split_set="train", task="proof_generation_all", open_world_assumption=True)
val_dataset = ProofWriterDataset(dataset_name="depth-5", split_set="val", task="proof_generation_all", open_world_assumption=True)

print("======== QA Dataset ==========")
proofwriter_collate_fn = ProofWriterQACollator(pretrained_t5_tokenizer="google/t5-v1_1-small")

train_dataloader = DataLoader(
    train_dataset, batch_size=2, collate_fn=proofwriter_collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=2, collate_fn=proofwriter_collate_fn
)

for batch in train_dataloader:
    print(batch)
    break

print("======== Proof Generation Dataset ==========")
proofwriter_collate_fn = ProofWriterProofGenerationAllCollator(pretrained_t5_tokenizer="google/t5-v1_1-small")
train_dataloader = DataLoader(
    train_dataset, batch_size=10, collate_fn=proofwriter_collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=10, collate_fn=proofwriter_collate_fn
)

for batch in train_dataloader:
    print(batch)
    break