import os
import wandb
import torch
import random
import importlib
import numpy as np
import pandas as pd 
import multiprocessing
from dotenv import load_dotenv
from datasets import load_dataset
from utils.metric import compute_metrics
from utils.encoder import Encoder
from utils.collator import DataCollatorForSimilarity
from utils.preprocessor import AnnotationRemover, BlankRemover
from utils.normalizer import Normalizer
from arguments import (ModelArguments, 
    DataTrainingArguments, 
    MyTrainingArguments, 
    LoggingArguments
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
        (ModelArguments, DataTrainingArguments, MyTrainingArguments, LoggingArguments)
    )
    # transformers.logging.set_verbosity_error()
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)

    load_dotenv(dotenv_path=logging_args.dotenv_path)
    POOLC_AUTH_KEY = os.getenv("POOLC_AUTH_KEY")   
    # -- Loading datasets
    # dset = load_dataset('sh110495/code-similarity', use_auth_token=HUGGINGFACE_AUTH_KEY)
    dset = load_dataset("PoolC/clone-det-base", use_auth_token=POOLC_AUTH_KEY)
    print(dset)

    CPU_COUNT = multiprocessing.cpu_count() // 2

    # -- Preprocessing datasets
    annotation_processor = AnnotationRemover()
    black_processor = BlankRemover()
    normalizer = Normalizer()
    dset = dset.map(annotation_processor, batched=True, num_proc=CPU_COUNT)
    dset = dset.map(normalizer, batched=True, num_proc=CPU_COUNT)
    dset = dset.map(black_processor, batched=True, num_proc=CPU_COUNT)
    print(dset)
    
    SIMILAR_FLAG = training_args.similarity_flag

    # -- Tokenizing & Encoding
    tokenizer = AutoTokenizer.from_pretrained(model_args.PLM)
    encoder = Encoder(tokenizer, similarlity_flag=SIMILAR_FLAG, max_input_length=data_args.max_length)
    dset = dset.map(encoder, batched=True, num_proc=CPU_COUNT, remove_columns=dset['train'].column_names)
    print(dset)

    # -- Config & Model Class
    config = AutoConfig.from_pretrained(model_args.PLM)
    config.num_labels = 2
    
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

    if training_args.do_train:

        if SIMILAR_FLAG :
            model = model_class(config=config)
            model.load_weight(model_args.PLM)
        else :
            model = model_class.from_pretrained(model_args.PLM, config=config)
         
        # -- Wandb
        WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
        wandb.login(key=WANDB_AUTH_KEY)

        group_name = model_args.PLM

        if SIMILAR_FLAG :
            group_name += '(similar model)'

        name = f"EP:{training_args.num_train_epochs}_LR:{training_args.learning_rate}_BS:{training_args.per_device_train_batch_size}_WR:{training_args.warmup_ratio}_WD:{training_args.weight_decay}_"
        name += MODEL_TYPE
        
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
            train_dataset=dset['train'],            # training dataset
            eval_dataset=dset['val'],        # evaluation dataset
            data_collator=data_collator,            # collator
            tokenizer=tokenizer,                    # tokenizer
            compute_metrics=compute_metrics,        # define metrics function
        )

        # -- Training
        trainer.train()
        trainer.evaluate()
        trainer.save_model(model_args.save_path)
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
