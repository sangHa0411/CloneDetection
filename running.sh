python train.py \
--do_train \
--fp16 \
--model_type rbert \
--logging_strategy steps \
--logging_steps 500 \
--evaluation_strategy steps \
--save_strategy steps \
--eval_steps 1000 \
--save_steps 1000 \
--overwrite_output_dir \
--save_total_limit 5 \
--output_dir ./exps \
--logging_dir ./logs \
--num_train_epochs 2 \
--learning_rate 2e-5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 2 \
--load_best_model_at_end \
--metric_for_best_model accuracy \
--warmup_ratio 0.05 \
--weight_decay 1e-2

# python inference.py \
# --tokenizer microsoft/codebert-base \
# --output_dir results \
# --file_name codebert_rbert_trainer_EP:2_BS:32_WR:0.05_WD:1e-2_LR:2e-5.csv \
# --PLM ./checkpoints \
# --fp16 \
# --per_device_eval_batch_size 32
