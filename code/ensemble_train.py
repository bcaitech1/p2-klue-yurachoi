import pickle as pickle
import os
import pandas as pd
import random
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from dataset import *

import argparse
from importlib import import_module
from pathlib import Path
import glob
import re

# import wandb

from loss import create_criterion
# os.environ['WANDB_WATCH'] = 'all'


class LabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        criterion = create_criterion('label_smoothing') # Default: CrossEntropyLoss
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = criterion
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

    
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.determiniztic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }


def increment_output_dir(output_path, exist_ok=False):
  path = Path(output_path)
  if (path.exists() and exist_ok) or (not path.exists()):
    return str(path)
  else:
    dirs = glob.glob(f"{path}*")
    matches = [re.search(rf"%s(\d+)" %path.stem, d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]
    n = max(i) + 1 if i else 2
    return f"{path}{n}"


def train(args):
  MY_TRAIN_TASKS = ["ForSemanticAnalysis"]
  seed_everything(args.seed)

  # Define tokniezer
  if args.model_task in MY_TRAIN_TASKS:
    tokenizer, added_token_nums = init_tokenizer(args.pretrained_model, ["REL"])
  else:
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)


  output_dir = increment_output_dir(args.output_dir)
  model_output_dir = os.path.join(output_dir, 'model')


  # Stratified K-Fold train-val split dataset
  # stratified_kfolds = StratifiedKFold(n_splits=args.n_folds, random_state=args.seed)
  stratified_kfolds = StratifiedKFold(n_splits=args.n_folds)
  
  dataset = load_pd_data("/opt/ml/input/data/train/train.tsv")
  # dataset = load_pd_data_2("/opt/ml/input/data/train/train_gold.tsv")

  labels = dataset['label'].values

  for train_idx, val_idx in stratified_kfolds.split(dataset, labels):
    train_dataset = dataset.iloc[train_idx]
    train_labels = train_dataset['label'].values
    val_dataset = dataset.iloc[val_idx]
    val_labels = val_dataset['label'].values
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_val = tokenize_dataset(val_dataset, tokenizer)
    train_dataset = RE_Dataset(tokenized_train, train_labels)
    val_dataset = RE_Dataset(tokenized_val, val_labels)
    train_model(args, train_dataset, val_dataset, model_output_dir)



def train_model(args, train_dataset, val_dataset, model_output_dir):
  MY_TRAIN_TASKS = ["ForSemanticAnalysis"]

  model_output_dir = increment_output_dir(model_output_dir)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter
  config_module = getattr(import_module("transformers"), args.model_type + "Config")
  model_config = config_module.from_pretrained(args.pretrained_model)
  model_config.num_labels = args.num_labels
  
  if args.model_task in MY_TRAIN_TASKS:
    model_module = getattr(import_module("models"), args.model_type + args.model_task)
    model = model_module.from_pretrained(args.pretrained_model, config=model_config, resize_vocab_num=tokenizer.vocab_size + added_token_nums)
  else:
    model_module = getattr(import_module("transformers"), args.model_type + args.model_task)
    model = model_module.from_pretrained(args.pretrained_model, config=model_config)
  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir=model_output_dir,          # output directory
    save_total_limit=args.save_total_limit,              # number of total save model.
    save_steps=args.save_steps,                 # model saving step.
    num_train_epochs=args.epochs,              # total number of training epochs
    learning_rate=args.lr,               # learning_rate
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    #per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
#     report_to='wandb',
    load_best_model_at_end = True,
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
  )
  
  if args.criterion == "LabelSmoothingLoss":
    trainer = LabelTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
    )
    
  else: 
    trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
    )

  # train model
  trainer.train()
  trainer.save_model(model_output_dir)
  trainer.save_state()
  
  del model
  # gc.collect()        
  torch.cuda.empty_cache()
    
    

def main(args):
  train(args)
#   wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='XLMRoberta')
    parser.add_argument('--model_task', type=str, default='ForSequenceClassification')
    parser.add_argument('--pretrained_model', type=str, default='xlm-roberta-large')
    parser.add_argument('--num_labels', type=int, default=42)
    
    parser.add_argument('--n_folds', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=500)               # number of warmup steps for learning rate scheduler
    parser.add_argument('--output_dir', type=str, default='./results/expr')
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--logging_dir', type=str, default='./logs')            # directory for storing logs
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--criterion', type=str, default="CrossEntropyLoss")


    args = parser.parse_args()
    
    main(args)