from typing import Optional
from dataclasses import dataclass, field
from model import SimilarityOutput
from transformers import TrainingArguments

@dataclass
class ModelArguments : 
    PLM: str = field(
        default="microsoft/codebert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    save_path: str = field(
        default="./checkpoints",
        metadata={
            "help": "Path to save checkpoint from fine tune model"
        },
    )
    
@dataclass
class DataTrainingArguments:
    max_length: int = field(
        default=512,
        metadata={
            "help": "Max length of input sequence"
        },
    )
    date_path: str = field(
        default='./data',
        metadata={
            "help": "Data directory"
        }
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={
            "help": "The number of preprocessing workers"
        }
    )
    
@dataclass
class MyTrainingArguments(TrainingArguments):
    report_to: Optional[str] = field(
        default='wandb',
    )
    fold_size : Optional[int] = field(
        default=5,
        metadata={"help" : "The number of folds"}
    )
    similarity_flag : Optional[bool] = field(
        default=False,
        metadata={"help" : "using cosing similarity model"}
    )
    model_type: Optional[str] = field(
        default='base',
        metadata={"help" : "model type"}
    )

@dataclass
class LoggingArguments:
    dotenv_path: Optional[str] = field(
        default='./wandb.env',
        metadata={"help":'input your dotenv path'},
    )
    project_name: Optional[str] = field(
        default="Code Similarity Checker",
        metadata={"help": "project name"},
    )

@dataclass
class InferenceArguments:
    file_name : Optional[str] = field(
        default='base_5fold.csv',
        metadata={"help" : "The csv file for test dataset"}
    )