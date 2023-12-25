#!/bin/bash
python training.py --output_dir outputs/lora_test_run \
--use_lora \
--device gpu \
--max_source_length 512 \
--max_target_length 512 \
--train_batch_size 32 \
--model_name_or_path google/flan-t5-large \
--gradient_accumulation_steps 1 \