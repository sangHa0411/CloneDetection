import os
import wandb
import torch
import random
import importlib
import numpy as np
import pandas as pd 
from datasets import Dataset
from utils.encoder import Encoder
from utils.preprocessor import Preprocessor
from utils.collator import DataCollatorForSimilarity
from tqdm import tqdm

from arguments import (ModelArguments, 
    DataTrainingArguments, 
    MyTrainingArguments, 
    InferenceArguments
)

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments, InferenceArguments)
    )
    model_args, data_args, training_args, inference_args = parser.parse_args_into_dataclasses()

    # -- Loading datasets
    df = pd.read_csv(os.path.join(data_args.date_path, 'test.csv'))
    
    # -- Preprocessing datasets
    preprocessor = Preprocessor()
    df['code1'] = df['code1'].apply(preprocessor)
    df['code2'] = df['code2'].apply(preprocessor)
    dset = Dataset.from_pandas(df)
    print(dset)

    SIMILAR_FLAG = training_args.similarity_flag

    # -- Tokenizing & Encoding
    tokenizer = AutoTokenizer.from_pretrained(inference_args.tokenizer)
    encoder = Encoder(tokenizer, SIMILAR_FLAG, data_args.max_length)
    dset = dset.map(encoder, batched=True, num_proc=4, remove_columns=dset.column_names)
    print(dset)

    # -- Model Class
    if SIMILAR_FLAG :
        model_lib = importlib.import_module('models.similar')
        model_class = getattr(model_lib, 'RobertaForSimilarityClassification')
    else :
        MODEL_TYPE = training_args.model_type
        if MODEL_TYPE == 'base' :
            model_class = AutoModelForSequenceClassification
        else :
            model_lib = importlib.import_module('models.base')
            if MODEL_TYPE == 'rbert' :
                model_class = getattr(model_lib, 'RobertaRBERT')
            else :
                assert NotImplementedError('Not Implemented Model type')

    # -- Collator
    collator_class = DataCollatorForSimilarity if SIMILAR_FLAG else DataCollatorWithPadding
    data_collator = collator_class(tokenizer=tokenizer, max_length=data_args.max_length)
    
    # -- Inference
    pred_probs = []
    for i in tqdm(range(training_args.fold_size)) :
        PLM = os.path.join(model_args.PLM, f'fold{i}')

        # -- Config & Model
        config = AutoConfig.from_pretrained(PLM)
        if SIMILAR_FLAG :
            model = model_class(inference_args.tokenizer, config=config)
            model.load_state_dict(torch.load(os.path.join(PLM, 'pytorch_model.bin')))
        else :
            model = model_class.from_pretrained(model_args.PLM, config=config)

        trainer = Trainer(                       # the instantiated 🤗 Transformers model to be trained
            model=model,                         # trained model
            args=training_args,                  # training arguments, defined above
            data_collator=data_collator,         # collator
        )

        # -- Inference
        outputs = trainer.predict(dset)
        pred_probs.append(outputs[0])

    pred = np.mean(pred_probs, axis=0)
    pred_ids = np.argmax(pred, axis=-1)
    sub_df = pd.read_csv(os.path.join(data_args.date_path, 'sample_submission.csv'))
    sub_df['similar'] = pred_ids
    sub_df.to_csv(os.path.join(training_args.output_dir, inference_args.file_name), index=False)

if __name__ == "__main__" :
    main()