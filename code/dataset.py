import numpy as np
import pandas as pd
import pickle

from transformers import AutoTokenizer
import torch

from sklearn.model_selection import StratifiedKFold



def init_tokenizer(pretrained_model, additional_tokens=[]):
    """
    Function that init tokenizer, which loads proper tokenizer and add tokens in given list of tokens
    - pretrained_model: Tokenizer pretrained model
    - additional_tokens: Tokens to add to the tokenizer

    - returns: tokenizer & number of additional tokens
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    if additional_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': additional_tokens})
    return tokenizer, len(additional_tokens)


# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenize_dataset(dataset, tokenizer):
    concat_entity = []
    for e01, e02 in zip(dataset['entity1'], dataset['entity2']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['text']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,
        add_special_tokens=True,
        )
    return tokenized_sentences




def tokenize_rel_dataset(dataset, tokenizer):
    rel_masked_sentences = []
    char1_indices = []
    char2_indices = []
    rel1_indices = [] # 토큰화 되었을 때
    rel2_indices = [] # 토큰화 되었을 때
    for text, beg1, end1, beg2, end2 in zip(dataset['text'], dataset['beg1'], dataset['end1'], dataset['beg2'], dataset['end2']):
        masked_text, token1_loc, token2_loc = mask_special_token(text, beg1, end1, beg2, end2)
        # tokenize_check(text, beg1, end1, beg2, end2, tokenizer)
        rel_masked_sentences.append(masked_text)
        char1_indices.append(token1_loc)
        char2_indices.append(token2_loc)
    
    tokenized_sentences = tokenizer(
        list(rel_masked_sentences),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        )
    
    rel1_indices = [tokenized_sentences.char_to_token(batch_or_char_index=i, char_index=char1) for i, char1 in enumerate(char1_indices)]
    rel2_indices = [tokenized_sentences.char_to_token(batch_or_char_index=i, char_index=char2) for i, char2 in enumerate(char2_indices)]

    return tokenized_sentences, rel1_indices, rel2_indices


# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Semantic relationship task를 위한 데이터셋
class RelationDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, ent1_locs, ent2_locs, labels):
        self.tokenized_dataset = tokenized_dataset
        self.ent1_locs = ent1_locs
        self.ent2_locs = ent2_locs
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['entity1_ids'] = torch.tensor(self.ent1_locs[idx])
        item['entity2_ids'] = torch.tensor(self.ent2_locs[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_pd_data(dataset_dir)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test = tokenize_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label


def load_semantic_testset(dataset_dir, tokenizer):
    test_dataset = load_pd_data(dataset_dir)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test, ent1_locs, ent2_locs = tokenize_rel_dataset(test_dataset, tokenizer)
    return tokenized_test, ent1_locs, ent2_locs, test_label



def load_pd_data(data_dir):
    """
    학습 데이터를 pandas DataFrmae 형태로 읽어들이고, label을 id로 변환한다
    - data_dir: train tsv 파일의 경로
    """
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(data_dir, delimiter='\t', names=["documentID", "text", "entity1", "beg1", "ent1_end", "entity2", "ent2_start", "ent2_end", "label"])
    label = []
    for i in dataset['label']:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    dataset['label'] = label
    return dataset


def load_pd_data_2(data_dir):
    """
    학습 데이터를 pandas DataFrmae 형태로 읽어들인다. 이미 label -> id 처리와 column name이 지정된 것을
    읽어올 때 사용하는 함수이다
    - data_dir: train tsv 파일의 경로
    """
    dataset = pd.read_csv(data_dir, delimiter='\t')
    dataset["label"].value_counts()
    return dataset


def mask_special_token(text, start1, end1, start2, end2):
    """
    원본 데이터에서 엔티티가 되는 부분을 마스크로 처리하고, 처리된 두 곳의 위치를 반환하는 함수입니다
    """
    # [REL] for relationship token
    changed = False
    if start1 > start2:
        start1, start2 = start2, start1
        end1, end2 = end2, end1
        changed = True
    text = text[:start1] + '[REL]'+ text[end1+1:start2] + '[REL]' + text[end2+1:]
    new1 = start1
    new2 = start2 - (end1 - start1) + 4
    if changed: new1, new2 = new2, new1
    return text, new1, new2


# TODO: 이거 원래대로 돌려놓기
# Token화 된 전후로 [REL] 토큰의 위치를 잘 찾고 있는지 확인하기 위한 함수. load_data.ipynb 확인
# 일단 [REL] 이 추가되었고 토큰 id가 35000이 되므로 char_to_token으로 비교한 것이 잘 되는지 확인함
def tokenize_check(sample_result, ent1_locs, ent2_locs, tokenizer, check_token_id=35000):
    """
    Special token을 추가하고 만든 문장에서, 해당 special token의 위치를 잘 잡았는지 확인하기 위함
    0 - text ([REL] 토큰 추가된 텍스트)
    1 - 첫째 REL 토큰의 원본텍스트에서의 시작
    2 - 둘째 REL 토큰의 원본텍스트에서의 시작
    """
    for i in range(len(ent1_locs)):
        tokenized_sentence = sample_result.input_ids[i]
        tokenized_index_1 = sample_result.char_to_token(i, ent1_locs[i])
        tokenized_index_2 = sample_result.char_to_token(i, ent2_locs[i])
        print(tokenized_index_1, tokenized_sentence[tokenized_index_1])
        print(tokenized_index_2, tokenized_sentence[tokenized_index_2])
        # print(tokenizer.decode(tokenized_sentence), tokenized_index_1, tokenized_index_2, tokenized_sentence[tokenized_index_1], tokenized_sentence[tokenized_index_2])
        assert (tokenized_sentence[tokenized_index_1] == check_token_id) and (tokenized_sentence[tokenized_index_2] == check_token_id)
