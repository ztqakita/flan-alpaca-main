#!/bin/bash
export CUDA_HOME=/usr/local/cuda-11.8
export TRANSFORMERS_OFFLINE=1

python training.py --output_dir outputs/lora_test_run \
--use_lora \
--device gpu \
--max_source_length 512 \
--max_target_length 512 \
--train_batch_size 2 \
--num_workers 18 \
--model_name_or_path /home/wusi_pkuhpc/.cache/huggingface/hub/models--google--flan-t5-large \
--gradient_accumulation_steps 16 \
