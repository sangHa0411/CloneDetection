
python inference.py \
--model_category t5 \
--model_name VHT5EncoderForSequenceClassification \
--output_dir results \
--file_name VHT5_EP:3_BS:48_WR:0.05_WD:1e-2_LR:3e-5_fold1_30000step.csv \
--PLM  /content/drive/MyDrive/1_fold_VHT5EncoderForSequenceClassification/checkpoint-30000 \
--per_device_eval_batch_size 48

python inference.py \
--model_category t5 \
--model_name VHT5EncoderForSequenceClassification \
--output_dir results \
--file_name VHT5_EP:3_BS:48_WR:0.05_WD:1e-2_LR:3e-5_fold1_36000step.csv \
--PLM  /content/drive/MyDrive/1_fold_VHT5EncoderForSequenceClassification/checkpoint-36000 \
--per_device_eval_batch_size 48

python inference.py \
--model_category t5 \
--model_name VHT5EncoderForSequenceClassification \
--output_dir results \
--file_name VHT5_EP:3_BS:48_WR:0.05_WD:1e-2_LR:3e-5_fold2_12000step.csv \
--PLM  /content/drive/MyDrive/1_fold_VHT5EncoderForSequenceClassification/checkpoint-12000 \
--per_device_eval_batch_size 48

python inference.py \
--model_category t5 \
--model_name VHT5EncoderForSequenceClassification \
--output_dir results \
--file_name VHT5_EP:3_BS:48_WR:0.05_WD:1e-2_LR:3e-5_fold2_31500step.csv \
--PLM  /content/drive/MyDrive/1_fold_VHT5EncoderForSequenceClassification/checkpoint-31500 \
--per_device_eval_batch_size 48


python inference.py \
--model_category t5 \
--model_name VHT5EncoderForSequenceClassification \
--output_dir results \
--file_name VHT5_EP:3_BS:48_WR:0.05_WD:1e-2_LR:3e-5_fold3_15000step.csv \
--PLM  /content/drive/MyDrive/1_fold_VHT5EncoderForSequenceClassification/checkpoint-15000 \
--per_device_eval_batch_size 48