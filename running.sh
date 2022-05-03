python train.py \
--do_train \
--fp16 \
--logging_strategy steps \
--logging_steps 200 \
--evaluation_strategy steps \
--save_strategy steps \
--eval_steps 1000 \
--save_steps 1000 \
--overwrite_output_dir \
--save_total_limit 3 \
--output_dir ./exps \
--logging_dir ./logs \
--fold_size 5 \
--num_train_epochs 3 \
--learning_rate 3e-5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--warmup_ratio 0.05 \
--weight_decay 1e-3

# python inference.py \
# --output_dir results \
# --file_name base_5fold_soft.csv \
# --fold_size 5 \
# --PLM ./checkpoints \
# --per_device_eval_batch_size 16