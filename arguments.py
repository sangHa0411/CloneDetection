from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    PLM: str = field(
        default="microsoft/codebert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    save_path: str = field(
        default="checkpoints", metadata={"help": "Path to save checkpoint from fine tune model"},
    )


@dataclass
class DataTrainingArguments:
    max_length: int = field(
        default=512, metadata={"help": "Max length of input sequence"},
    )
    date_path: str = field(default="data", metadata={"help": "Data directory"})
    do_all: bool = field(default=False, metadata={"help": "Kfold cross validation: False, All Training: True"})


@dataclass
class MyTrainingArguments(TrainingArguments):
    report_to: Optional[str] = field(default="wandb",)
    model_name: Optional[str] = field(
        default="base",
        metadata={
            "help": "model class if class is base, it returns AutoModelForSequenceClassification class"
        },
    )
    model_category: Optional[str] = field(
        default="plbart", metadata={"help": "model category (plbart, t5, codebert)"}
    )


@dataclass
class LoggingArguments:
    dotenv_path: Optional[str] = field(
        default="wandb.env", metadata={"help": "input your dotenv path"},
    )
    project_name: Optional[str] = field(
        default="Dacon - Clone Detction", metadata={"help": "project name"},
    )


@dataclass
class InferenceArguments:
    file_name: Optional[str] = field(
        default="base.csv", metadata={"help": "The csv file for test dataset"}
    )
