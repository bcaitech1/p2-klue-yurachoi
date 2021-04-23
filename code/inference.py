from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
# from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
from pathlib import Path
import os
import re

# import argparse
from importlib import import_module
from dataset import *

def save_logits(model, tokenized_sent, device, args):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  output_pred = []
  model.eval()
  
  for i, data in enumerate(dataloader):
    print(data)
    with torch.no_grad():
      if args.model_task == "ForSemanticAnalysis":
        outputs = model(
            input_ids=data['input_ids'].to(device),
            attention_mask=data['attention_mask'].to(device),
            token_type_ids=data['token_type_ids'].to(device),
            entity1_ids=data['entity1_ids'].to(device), # 내가 추가한 인자
            entity2_ids=data['entity2_ids'].to(device), # 내가 추가한 인자
            )
      else:
        outputs = model(
            input_ids=data['input_ids'].to(device),
            attention_mask=data['attention_mask'].to(device),
            )
    logits = outputs['logits']
    logits = logits.detach().cpu().numpy()
    print(logits.shape)
    for logit in logits:
      output_pred.append(logit)
  pred_logits = np.array(output_pred)
  print(len(pred_logits))
  output = pd.DataFrame(pred_logits, columns=[i for i in range(42)])
  output.to_csv(args.out_path, index=False)


def inference(model, tokenized_sent, device, args):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  output_pred = []
  model.eval()
  
  for i, data in enumerate(dataloader):
    print(data)
    with torch.no_grad():
      if args.model_task == "ForSemanticAnalysis":
        outputs = model(
            input_ids=data['input_ids'].to(device),
            attention_mask=data['attention_mask'].to(device),
            token_type_ids=data['token_type_ids'].to(device),
            entity1_ids=data['entity1_ids'].to(device), # 내가 추가한 인자
            entity2_ids=data['entity2_ids'].to(device), # 내가 추가한 인자
            )
      else:
        if args.model_type == "XLMRoberta":
          outputs = model(
              input_ids=data['input_ids'].to(device),
              attention_mask=data['attention_mask'].to(device),
              )
        else:
          outputs = model(
              input_ids=data['input_ids'].to(device),
              attention_mask=data['attention_mask'].to(device),
              token_type_ids=data['token_type_ids'].to(device),
              )
    logits = outputs['logits']
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
  
  pred_answer = np.array(output_pred).flatten()
  output = pd.DataFrame(pred_answer, columns=['pred'])
  output.to_csv(args.out_path, index=False)
  # return output


def main(args):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  MY_TRAIN_TASKS = ["ForSemanticAnalysis"]
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  # load tokenizer
  TOK_NAME = args.pretrained_model
  if args.model_task in MY_TRAIN_TASKS:
    tokenizer, added_token_nums = init_tokenizer(TOK_NAME, ["[REL]"])
  else:
    tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)
  

  # load my model
  if args.model_task in MY_TRAIN_TASKS:
    model_module = getattr(import_module("models"), args.model_type + args.model_task)
    model = model_module.from_pretrained(args.model_dir, resize_vocab_num=tokenizer.vocab_size + added_token_nums)
  else:
    model_module = getattr(import_module("transformers"), args.model_type + "ForSequenceClassification")
    model = model_module.from_pretrained(args.model_dir)
  model.parameters
  model.to(device)

  # load test datset
  test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
  if args.model_task in MY_TRAIN_TASKS:
    test_dataset, ent1_locs, ent2_locs, test_label = load_semantic_testset(test_dataset_dir, tokenizer)
    # make dataset for pytorch
    test_dataset = RelationDataset(tokenized_test, ent1_locs, ent2_locs, test_label)
  else:
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    # make dataset for pytorch
    test_dataset = RE_Dataset(test_dataset ,test_label)


  # predict answer
  if args.inference_type == "labels":
    inference(model, test_dataset, device, args)
  elif args.inference_type == "logits":
    save_logits(model, test_dataset, device, args)
  else:
    print("Invalid inference type")

  # make csv file with predicted answer
  print("Inference done!")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # /opt/ml/code/results/xlm-roberta1/checkpoint-1000/config.json
  # /opt/ml/code/results/koelectra_semantic/checkpoint-5500
  # /opt/ml/code/results/fold_expr_1/xlm_seq_1/checkpoint-1000
  # /opt/ml/code/results/expr15/checkpoint-1500/config.json
  # /opt/ml/code/results/expr8/checkpoint-5500/config.json
  # /opt/ml/code/results/expr7/checkpoint-2000/config.json

  # /opt/ml/code/results/ensemble_final7/model[0-9]*
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="./results/expr/checkpoint-2000")
  parser.add_argument('--out_path', type=str, default="./prediction/submission.csv")
  parser.add_argument('--model_type', type=str, default="Bert")
  parser.add_argument('--model_task', type=str, default="ForSequenceClassification")
  parser.add_argument('--pretrained_model', type=str, default="bert-base-multilingual-cased")
  parser.add_argument('--inference_type', type=str, default="labels")
  args = parser.parse_args()
  print(args)
  main(args)
  
