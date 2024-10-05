import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainerCallback,
    AdamW,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)
import math
import os
import torch


class CosineAnnealingScheduler(TrainerCallback):
    """Custom LR Scheduler that implements a cosine annealing schedule with warmup."""

    def __init__(
        self,
        optimizer,
        num_warmup_steps,
        num_training_steps,
        num_cycles=0.5,
        last_epoch=-1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.last_epoch = last_epoch
        self.optimizer = optimizer

    def on_step_begin(self, args, state, control, **kwargs):
        """Called right before a training step in the main training loop."""
        step = state.global_step
        if step < self.num_warmup_steps:
            lr_scale = float(step) / float(max(1, self.num_warmup_steps))
        else:
            progress = float(step - self.num_warmup_steps) / float(
                max(1, self.num_training_steps - self.num_warmup_steps)
            )
            lr_scale = max(
                0.0, 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress))
            )

        for group in self.optimizer.param_groups:
            group["lr"] = lr_scale * group["initial_lr"]


class MetricsCallback(TrainerCallback):
    "A callback that stores all intermediate training, validation losses and validation accuracy."

    def __init__(self):
        super().__init__()
        self.training_losses = []
        self.validation_losses = []
        self.validation_accuracy = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Logs appear as a dictionary. Check if loss and eval_loss are in the dictionary and append them.
        if "loss" in logs:
            self.training_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.validation_losses.append(logs["eval_loss"])
        if "eval_accuracy" in logs:
            self.validation_accuracy.append(logs["eval_accuracy"])


def preprocess_function(examples):
    # Assuming 'text' and 'label' are column names in your dataset
    result = tokenizer(
        examples["full_text"], padding="max_length", truncation=True, max_length=512
    )
    result["labels"] = examples["labels"]
    return result


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels.flatten(), predictions.flatten())
    kappa = cohen_kappa_score(
        labels.flatten(), predictions.flatten(), weights="quadratic"
    )
    return {"accuracy": accuracy, "kappa": kappa}


data = pd.read_csv("./data/final_data/train.csv")
data = data.rename(columns={"score": "labels"})

data["labels"] = data["labels"] - 1
num_classes = data["labels"].nunique()

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

dataset = Dataset.from_pandas(data.iloc[:, 1:])
tokenized_dataset = dataset.map(preprocess_function, batched=True)

split_datasets = tokenized_dataset.train_test_split(
    test_size=0.2, shuffle=True, seed=46
)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

num_epochs = 7

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # evaluation is done at the end of each epoch
    save_strategy="epoch",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="kappa",
)

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=num_classes
)

optimizer = AdamW(model.parameters(), lr=1e-5)
num_training_steps = (
    num_epochs * len(train_dataset) // training_args.per_device_train_batch_size
)
scheduler_callback = CosineAnnealingScheduler(
    optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
)
metrics_callback = MetricsCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    optimizers=(optimizer, None),
    compute_metrics=compute_metrics,
    callbacks=[scheduler_callback, metrics_callback],
)

trainer.train()

import boto3
from dotenv import load_dotenv

cwd = os.getcwd()
load_dotenv(dotenv_path=os.path.join("config", ".env"))

# Set up AWS credentials (make sure you have the necessary permissions)
session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

s3_client = session.client("s3")

# Define the S3 bucket and key for the model
s3_bucket = os.getenv("S3_BUCKET")

s3_model_dir_path = os.getenv("S3_MODEL_DIR_PATH")
s3_tokenizer_dir_path = os.getenv("S3_TOKENIZER_DIR_PATH")

local_model_dir_path = os.getenv("LOCAL_MODEL_DIR_PATH")
local_tokenizer_dir_path = os.getenv("LOCAL_TOKENIZER_DIR_PATH")

file_type = ".zip"
file_action = "zip"

model_name = "deBERTa-v3"

dummy_input = tokenizer(
    "This is a dummy input",
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512,
)

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    f"{cwd}{local_model_dir_path}{model_name}",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
)


tokenizer.save_pretrained(f"{cwd}{local_tokenizer_dir_path}{model_name}")
os.system(
    f"{file_action} -r {cwd}{local_tokenizer_dir_path}{model_name}{file_type} {cwd}{local_tokenizer_dir_path}{model_name}"
)
s3_client.upload_file(
    f"{cwd}{local_tokenizer_dir_path}", s3_bucket, f"{model_name}{file_type}"
)
os.system(f"rm {cwd}{local_tokenizer_dir_path}{model_name}{file_type}")

trainer.save_pretrained(f"{cwd}{local_model_dir_path}{model_name}")
os.system(
    f"{file_action} -r {cwd}{local_model_dir_path}{model_name}{file_type} {cwd}{local_model_dir_path}{model_name}"
)
s3_client.upload_file(
    f"{cwd}{local_model_dir_path}", s3_bucket, f"{model_name}{file_type}"
)
os.system(f"rm {cwd}{local_model_dir_path}{model_name}{file_type}")
