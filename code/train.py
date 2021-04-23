import pickle as pickle
import os
import pandas as pd
import random
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from load_data import *
from dataset import *

import argparse
from importlib import import_module
from pathlib import Path
import glob
import re

import wandb
os.environ['WANDB_WATCH'] = 'all'


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
    matches = [re.search(rf"%s(\d+)" %path.stem, d) for d in dirs] # path.stem - 가장 하위 디렉토리
    i = [int(m.groups()[0]) for m in matches if m]
    n = max(i) + 1 if i else 2
    return f"{path}{n}"


def train(args):
  # 직접 정의한 방식들로 학습을 시킬 때
  seed_everything(args.seed)
  MY_TRAIN_TASKS = ["ForSemanticAnalysis", "ForSemanticAnalysis2"]

  # load model and tokenizer
  MODEL_NAME = args.pretrained_model
  if args.model_task == "ForSemanticAnalysis":
    tokenizer, added_token_nums = init_tokenizer(MODEL_NAME, ["REL"])
  else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  dataset = load_pd_data_2(f"/opt/ml/input/data/train/train_gold_kfold_{args.fold_idx}.tsv")
  train_dataset = dataset.loc[dataset["train_val"] == 0]
  dev_dataset = dataset.loc[dataset["train_val"] == 1]
  print("Train dataset length: ", len(train_dataset))
  print("Validation dataset length: ", len(dev_dataset))

  train_label = train_dataset['label'].values
  dev_label = dev_dataset['label'].values
  
  # tokenizing dataset
  if args.model_task == "ForSemanticAnalysis":
    tokenized_train, train_ent1_locs, train_ent2_locs = tokenize_rel_dataset(train_dataset, tokenizer)
    tokenized_dev, dev_ent1_locs, dev_ent2_locs = tokenize_rel_dataset(train_dataset, tokenizer)
  else:
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenize_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  if args.model_task == "ForSemanticAnalysis":
    train_dataset = RelationDataset(tokenized_train, train_ent1_locs, train_ent2_locs, train_label)
    dev_dataset = RelationDataset(tokenized_dev, dev_ent1_locs, dev_ent2_locs, dev_label)
  else:
    train_dataset = RE_Dataset(tokenized_train, train_label)
    dev_dataset = RE_Dataset(tokenized_dev, dev_label)


  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter
  config_module = getattr(import_module("transformers"), args.model_type + "Config")
  model_config = config_module.from_pretrained(MODEL_NAME)
  model_config.num_labels = args.num_labels
  
  if args.model_task in MY_TRAIN_TASKS:
    model_module = getattr(import_module("models"), args.model_type + args.model_task)
    if args.model_task == "ForSemanticAnalysis":
      model = model_module.from_pretrained(MODEL_NAME, config=model_config, resize_vocab_num=tokenizer.vocab_size + added_token_nums, pretrained_name=MODEL_NAME)
    else:
      model = model_module.from_pretrained(MODEL_NAME, config=model_config, pretrained_name=MODEL_NAME)
  else:
    model_module = getattr(import_module("transformers"), args.model_type + args.model_task)
    model = model_module.from_pretrained(MODEL_NAME, config=model_config)
  model.parameters
  model.to(device)

  output_dir = increment_output_dir(args.output_dir)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir=output_dir,          # output directory
    # save_total_limit=args.save_total_limit,              # number of total save model.
    # save_steps=args.save_steps,                 # model saving step.
    num_train_epochs=args.epochs,              # total number of training epochs
    learning_rate=args.lr,               # learning_rate
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    #per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    report_to='wandb',
    # load_best_model_at_end = True
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
  )
  
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()
  trainer.save_model(output_dir)
  trainer.save_state()
  # 학습이 끝나면 wandb 프로세스도 종료시킴
  wandb.finish()


def main(args):
  train(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='Bert')
    parser.add_argument('--model_task', type=str, default='ForSequenceClassification')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-multilingual-cased')
    parser.add_argument('--num_labels', type=int, default=42)
    
    parser.add_argument('--n_folds', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=500)               # number of warmup steps for learning rate scheduler
    parser.add_argument('--output_dir', type=str, default='./results/expr')
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--logging_dir', type=str, default='./logs')            # directory for storing logs
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--fold_idx', type=int, default=1)


    args = parser.parse_args()
    
    main(args)