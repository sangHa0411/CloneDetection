import os
import wandb
import torch
import random
import importlib
import numpy as np
import pandas as pd 
from dotenv import load_dotenv
from datasets import Dataset
from model import RobertaForSimilarityClassification
from utils.encoder import Encoder
from utils.metric import compute_metrics
from utils.collator import DataCollatorWithPadding
from utils.preprocessor import Preprocessor, BlockDeleter
from sklearn.model_selection import StratifiedKFold
from arguments import (ModelArguments, 
    DataTrainingArguments, 
    MyTrainingArguments, 
    LoggingArguments
)

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)

    # -- Loading datasets
    df = pd.read_csv(os.path.join(data_args.date_path, 'sample_train.csv'))
    
    # -- Preprocessing datasets
    preprocessor = Preprocessor()
    df['code1'] = df['code1'].apply(preprocessor)
    df['code2'] = df['code2'].apply(preprocessor)
    dset = Dataset.from_pandas(df)
    print(dset)

    # -- Tokenizing & Encoding
    tokenizer = AutoTokenizer.from_pretrained(model_args.PLM)
    encoder = Encoder(tokenizer, data_args.max_length)
    dset = dset.map(encoder, batched=True, num_proc=4, remove_columns=dset.column_names)
    print(dset)

    # -- Model Class
    config = AutoConfig.from_pretrained(model_args.PLM)
    config.num_labels = 1
    model_class = RobertaForSimilarityClassification
   
    # -- Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)

    if training_args.do_train:
        skf = StratifiedKFold(n_splits=training_args.fold_size, shuffle=True)

        for i, (train_idx, valid_idx) in enumerate(skf.split(dset, dset['labels'])):
            model = model_class(model_checkpoint=model_args.PLM, config=config)
            
            train_dataset = dset.select(train_idx.tolist())
            valid_dataset = dset.select(valid_idx.tolist())
                        
            # -- Wandb
            load_dotenv(dotenv_path=logging_args.dotenv_path)
            WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
            wandb.login(key=WANDB_AUTH_KEY)

            group_name = model_args.PLM + '-' + str(training_args.fold_size) + '-fold-training'
            name = f"EP:{training_args.num_train_epochs}\
                _LR:{training_args.learning_rate}\
                _BS:{training_args.per_device_train_batch_size}\
                _WR:{training_args.warmup_ratio}\
                _WD:{training_args.weight_decay}"
        
            wandb.init(
                entity="sangha0411",
                project=logging_args.project_name,
                group=group_name,
                name=name
            )
            wandb.config.update(training_args)

            trainer = Trainer(                          # the instantiated 🤗 Transformers model to be trained
                model=model,                            # model
                args=training_args,                     # training arguments, defined above
                train_dataset=train_dataset,            # training dataset
                eval_dataset=valid_dataset,             # evaluation dataset
                data_collator=data_collator,            # collator
                tokenizer=tokenizer,                    # tokenizer
                compute_metrics=compute_metrics,        # define metrics function
            )

            # -- Training
            trainer.train()
            save_path = os.path.join(model_args.save_path, f'fold{i}')
            trainer.evaluate()
            trainer.save_model(save_path)
            wandb.finish()  
            
            if training_args.do_eval:
                break
            
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)

if __name__ == '__main__':
    main()