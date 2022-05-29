import os
import wandb
import torch
import random
import numpy as np
import importlib
import multiprocessing
from dotenv import load_dotenv
from datasets import load_dataset, concatenate_datasets
from utils.metric import compute_metrics
from utils.encoder import Encoder
from utils.collator import DataCollatorWithPadding
from utils.preprocessor import AnnotationPreprocessor, FunctionPreprocessor
from sklearn.model_selection import StratifiedKFold
from trainer import Trainer
from arguments import (ModelArguments, 
    DataTrainingArguments, 
    MyTrainingArguments, 
    LoggingArguments
)

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)

    load_dotenv(dotenv_path=logging_args.dotenv_path)
    POOLC_AUTH_KEY = os.getenv("POOLC_AUTH_KEY")  

    # -- Loading datasets
    dset = load_dataset("PoolC/clone-det-base", use_auth_token=POOLC_AUTH_KEY)
    print(dset)

    dset = concatenate_datasets([dset['train'], dset['val']]).shuffle(training_args.seed)

    CPU_COUNT = multiprocessing.cpu_count() // 2

    MAX_LENGTH = 4000
    def filter_fn(data) :
        if len(data['code1']) >= MAX_LENGTH or len(data['code2']) >= MAX_LENGTH :
            return False
        else :
            return True

    dset = dset.filter(filter_fn, num_proc=CPU_COUNT)
    print(dset)

    # -- Preprocessing datasets
    fn_preprocessor = FunctionPreprocessor()
    dset = dset.map(fn_preprocessor, batched=True, num_proc=CPU_COUNT)

    an_preprocessor = AnnotationPreprocessor()
    dset = dset.map(an_preprocessor, batched=True, num_proc=CPU_COUNT)

    # -- Tokenizing & Encoding
    MODEL_CATEGORY = training_args.model_category

    tokenizer = AutoTokenizer.from_pretrained(model_args.PLM)
    encoder = Encoder(tokenizer, model_category=MODEL_CATEGORY, max_input_length=data_args.max_length)
    dset = dset.map(encoder, batched=True, num_proc=CPU_COUNT, remove_columns=dset.column_names)
    print(dset)

    # -- Config & Model Class
    config = AutoConfig.from_pretrained(model_args.PLM)
    config.num_labels = 2
    config.tokenizer_cls_token_id = tokenizer.cls_token_id
    config.tokenizer_sep_token_id = tokenizer.sep_token_id
    
    MODEL_NAME = training_args.model_name
    if MODEL_NAME == 'base' :
        model_class = AutoModelForSequenceClassification
    else :
        model_category = importlib.import_module('models.' + MODEL_CATEGORY)
        model_class = getattr(model_category, MODEL_NAME)
   
    # -- Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)

    if training_args.do_train:
        skf = StratifiedKFold(n_splits=training_args.fold_size, shuffle=True)
        for i, (train_idx, valid_idx) in enumerate(skf.split(dset, dset['labels'])):
            
            train_dataset = dset.select(train_idx.tolist())
            valid_dataset = dset.select(valid_idx.tolist())

            training_args.remove_unused_columns = False
            model = model_class.from_pretrained(model_args.PLM, config=config)
            
            # -- Wandb
            WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
            wandb.login(key=WANDB_AUTH_KEY)

            group_name = model_args.PLM + "_K-fold"

            if training_args.max_steps == -1 :
                name = f"EP:{training_args.num_train_epochs}_"
            else :
                name = f"MS:{training_args.max_steps}_"
                
            name += f"LR:{training_args.learning_rate}_BS:{training_args.per_device_train_batch_size}_WR:{training_args.warmup_ratio}_WD:{training_args.weight_decay}_"
            name += MODEL_NAME 
            name += f"_fold{i}"
            
            wandb.init(
                entity="poolc",
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

            save_path = os.path.join(model_args.save_path, f"fold_{i}")

            # -- Training
            trainer.train()
            trainer.evaluate()
            trainer.save_model(save_path)
            wandb.finish()  
        
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
