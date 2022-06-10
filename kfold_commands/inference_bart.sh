python inference.py \
--model_category plbart \
--model_name VHBartEncoderForSequenceClassification \
--output_dir results \
--file_name VHPLBART_EP:3_BS:32_WR:0.00_WD:0.00_LR:3e-5_fold3_28000step.csv \
--PLM /home/noopy/CodeSimilarity/exps/3_fold_VHBartEncoderForSequenceClassification/checkpoint-28000 \
--per_device_eval_batch_size 32

python inference.py \
--model_category plbart \
--model_name VHBartEncoderForSequenceClassification \
--output_dir results \
--file_name VHPLBART_EP:3_BS:32_WR:0.00_WD:0.00_LR:3e-5_fold3_44000step.csv \
--PLM /home/noopy/CodeSimilarity/exps/3_fold_VHBartEncoderForSequenceClassification/checkpoint-44000 \
--per_device_eval_batch_size 32


python inference.py \
--model_category plbart \
--model_name VHBartEncoderForSequenceClassification \
--output_dir results \
--file_name VHPLBART_EP:3_BS:32_WR:0.00_WD:0.00_LR:3e-5_fold3_46000step.csv \
--PLM /home/noopy/CodeSimilarity/exps/3_fold_VHBartEncoderForSequenceClassification/checkpoint-46000 \
--per_device_eval_batch_size 32

