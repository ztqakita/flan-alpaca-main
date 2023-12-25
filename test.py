import torch
from peft import LoraConfig, TaskType, get_peft_model, LoraModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

model_name = "google/t5-v1_1-small"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
     
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,

    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

# print(list(model.named_modules()))

key_list = [key for key, _ in model.named_modules()]

# print(key_list)

peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q", r"encoder\..*?\.k", "v"],
            )


model = get_peft_model(model, peft_config, "default")

# model.print_trainable_parameters()

# Printing trainable parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.size())