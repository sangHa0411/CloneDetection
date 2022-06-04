# train command example
python train.py \
--do_train \
--fp16 \
--PLM microsoft/graphcodebert-base \
--model_category codebert \
--model_name RobertaRBERT \
--logging_strategy steps \
--logging_steps 1000 \
--evaluation_strategy steps \
--save_strategy steps \
--eval_steps 2000 \
--save_steps 2000 \
--overwrite_output_dir \
--save_total_limit 10 \
--output_dir ./exps \
--logging_dir ./logs \
--num_train_epochs 2 \
--learning_rate 2e-5 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 2 \
--load_best_model_at_end \
--metric_for_best_model accuracy \
--warmup_ratio 0.05 \
--weight_decay 1e-2

### Fold 1
python inference.py \
--model_category codebert \
--model_name RobertaRBERT \
--output_dir results \
--file_name graphcodebert_rbert_EP:2_BS:32_WR:0.05_WD:1e-2_LR:2e-5_fold1.csv \
--PLM  /home/ubuntu/CodeSimilarity/exps/1_fold/checkpoint-16000 \
--fp16 \
--per_device_eval_batch_size 32

### Fold 2
python inference.py \
--model_category codebert \
--model_name RobertaRBERT \
--output_dir results \
--file_name graphcodebert_rbert_EP:2_BS:32_WR:0.05_WD:1e-2_LR:2e-5_fold2.csv \
--PLM  /home/ubuntu/CodeSimilarity/exps/2_fold/checkpoint-10000 \
--fp16 \
--per_device_eval_batch_size 32

### Fold 3
python inference.py \
--model_category codebert \
--model_name RobertaRBERT \
--output_dir results \
--file_name graphcodebert_rbert_EP:2_BS:32_WR:0.05_WD:1e-2_LR:2e-5_fold3.csv \
--PLM  /home/ubuntu/CodeSimilarity/exps/3_fold/checkpoint-16000 \
--fp16 \
--per_device_eval_batch_size 32

### Fold 4
python inference.py \
--model_category codebert \
--model_name RobertaRBERT \
--output_dir results \
--file_name graphcodebert_rbert_EP:2_BS:32_WR:0.05_WD:1e-2_LR:2e-5_fold4.csv \
--PLM  /home/ubuntu/CodeSimilarity/exps/4_fold/checkpoint-12000 \
--fp16 \
--per_device_eval_batch_size 32

### Fold 5
python inference.py \
--model_category codebert \
--model_name RobertaRBERT \
--output_dir results \
--file_name graphcodebert_rbert_EP:2_BS:32_WR:0.05_WD:1e-2_LR:2e-5_fold5.csv \
--PLM  /home/ubuntu/CodeSimilarity/exps/5_fold/checkpoint-8000 \
--fp16 \
--per_device_eval_batch_size 32
