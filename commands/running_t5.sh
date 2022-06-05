# train command example
python train.py \
--do_train \
--fp16 \
--PLM Salesforce/codet5-base \
--model_category t5 \
--model_name VHT5EncoderForSequenceClassification \
--logging_strategy steps \
--logging_steps 1500 \
--evaluation_strategy steps \
--save_strategy steps \
--eval_steps 1500 \
--save_steps 1500 \
--overwrite_output_dir \
--save_total_limit 10 \
--output_dir ./exps \
--logging_dir ./logs \
--num_train_epochs 2 \
--learning_rate 3e-5 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 48 \
--gradient_accumulation_steps 1 \
--metric_for_best_model accuracy \
--warmup_ratio 0.10 \
--weight_decay 1e-2

### Fold 1
python inference.py \
--fp16 \
--model_category t5 \
--model_name VHT5EncoderForSequenceClassification \
--output_dir results \
--file_name VHT5_EP:3_BS:32_WR:0.10_WD:1e-2_LR:3e-5_fold1.csv \
--PLM  /home/ubuntu/CodeSimilarity/exps/1_fold_VHT5EncoderForSequenceClassification/checkpoint-16000 \
--per_device_eval_batch_size 48