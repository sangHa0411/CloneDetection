import os
import importlib
import numpy as np
import pandas as pd 
import multiprocessing
from datasets import Dataset
from utils.encoder import Encoder
from utils.normalizer import Normalizer
from utils.collator import DataCollatorForSimilarity
from utils.preprocessor import Preprocessor

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
    dset = Dataset.from_pandas(df)
    print(dset)

    CPU_COUNT = multiprocessing.cpu_count() // 2

    # -- Preprocessing datasets
    preprocessor = Preprocessor()
    dset = dset.map(preprocessor, batched=True, num_proc=CPU_COUNT)
    print(dset)

    MAX_LENGTH = 1500
    def map_fn(data) :
        data['code1'] = data['code1'][:MAX_LENGTH]
        data['code2'] = data['code2'][:MAX_LENGTH]
        return data

    dset = dset.map(map_fn, batched=False, num_proc=CPU_COUNT)
    print(dset)

    normalizer = Normalizer()
    dset = dset.map(normalizer, batched=True, num_proc=CPU_COUNT)
    print(dset)

    SIMILAR_FLAG = training_args.similarity_flag

    # -- Tokenizing & Encoding
    tokenizer = AutoTokenizer.from_pretrained(inference_args.tokenizer)
    encoder = Encoder(tokenizer, SIMILAR_FLAG, data_args.max_length)
    dset = dset.map(encoder, batched=True, num_proc=multiprocessing.cpu_count(), remove_columns=dset.column_names)
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
    # -- Config & Model
    print(model_args.PLM)
    config = AutoConfig.from_pretrained(model_args.PLM)
    model = model_class.from_pretrained(model_args.PLM, config=config)

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