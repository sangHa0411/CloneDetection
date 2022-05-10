python train.py \
--do_train \
--fp16 \
--logging_strategy steps \
--logging_steps 100 \
--evaluation_strategy steps \
--save_strategy steps \
--eval_steps 2000 \
--save_steps 2000 \
--overwrite_output_dir \
--save_total_limit 5 \
--output_dir ./exps \
--logging_dir ./logs \
--num_train_epochs 2 \
--learning_rate 1e-5 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--warmup_ratio 0.05 \
--weight_decay 1e-2

# python inference.py \
# --tokenizer uclanlp/plbart-base \
# --output_dir results \
# --file_name EP:2_BS:16_WR:0.05_WD:1e-3_LR:3e-5_Fold:5_Soft.csv \
# --fold_size 5 \
# --PLM ./checkpoints \
# --per_device_eval_batch_size 32