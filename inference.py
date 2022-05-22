import os
import importlib
import numpy as np
import pandas as pd 
import multiprocessing
from datasets import Dataset
from trainer import Trainer
from utils.encoder import Encoder
from utils.preprocessor import Preprocessor
from utils.collator import DataCollatorWithPadding

from arguments import (ModelArguments, 
    DataTrainingArguments, 
    MyTrainingArguments, 
    InferenceArguments
)

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments, InferenceArguments)
    )
    model_args, data_args, training_args, inference_args = parser.parse_args_into_dataclasses()

    # -- Loading datasets
    # transformers.logging.set_verbosity_error()
    df = pd.read_csv(os.path.join(data_args.date_path, 'test.csv'))
    dset = Dataset.from_pandas(df)
    print(dset)

    CPU_COUNT = multiprocessing.cpu_count() // 2

    # -- Preprocessing datasets
    preprocessor = Preprocessor()
    dset = dset.map(preprocessor, batched=True, num_proc=CPU_COUNT)
    print(dset)

    # -- Tokenizing & Encoding
    MODEL_CATEGORY = training_args.model_category

    tokenizer = AutoTokenizer.from_pretrained(model_args.PLM)
    encoder = Encoder(tokenizer, model_category=MODEL_CATEGORY, max_input_length=data_args.max_length)
    dset = dset.map(encoder, batched=True, num_proc=multiprocessing.cpu_count(), remove_columns=dset.column_names)
    print(dset)

    # -- Model Class
    MODEL_NAME = training_args.model_name

    if MODEL_NAME == 'base' :
        model_class = AutoModelForSequenceClassification
    else :
        model_category = importlib.import_module('models.' + MODEL_CATEGORY)
        model_class = getattr(model_category, MODEL_NAME)
        
    # -- Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)
    
    # -- Inference
    # -- Config & Model
    config = AutoConfig.from_pretrained(model_args.PLM)
    model = model_class.from_pretrained(model_args.PLM, config=config)
    training_args.remove_unused_columns = False

    trainer = Trainer(                       # the instantiated 🤗 Transformers model to be trained
        model=model,                         # trained model
        args=training_args,                  # training arguments, defined above
        data_collator=data_collator,         # collator
    )

    # -- Inference
    outputs = trainer.predict(dset)

    pred_ids = np.argmax(outputs[0], axis=-1)
    sub_df = pd.read_csv(os.path.join(data_args.date_path, 'sample_submission.csv'))
    sub_df['similar'] = pred_ids
    
    sub_df.to_csv(os.path.join(training_args.output_dir, inference_args.file_name), index=False)

if __name__ == "__main__" :
    main()