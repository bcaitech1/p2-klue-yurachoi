# Pstage_02_KLUE_Relation_extraction

## Training
---
* python train.py


### Arguments
---


**모델 초기화 관련 (train.py, inference.py 공통)**

- `pretrained_model` : 사전학습된 모델입니다. 기본값은 bert-base-multilingual-cased 입니다.
- `model_type` : `<model_type>ForSequenceClassification` 모델 구조를 사용하고, `<model_type>Config` 를 인자로 받습니다. 기본값은 Bert입니다.

**학습 관련 하이퍼파라미터 (train.py)**


- `epochs` : 모델 학습 시킬 epoch 수입니다. 기본값: `4`
- `batch_size`: 장치 별 데이터 배치 크기입니다. 기본값: `16`
- lr: 학습률입니다. 기본값: `5e-5`
- `weight_decay`: 오버피팅을 방지하기 위해 weight가 큰 값을 갖지 않도록 주는 패널티의 크기입니다. 기본값: `0.01`
- `warmup_steps`: 학습률 스케줄러의 warm up 단계입니다. 현재 step이 warmup보다 낮을 경우 학습률을 선형적으로 증가시키고, 이후에는 각 스케줄러에서 정한 방법데로 학습률을 업데이트합니다. 기본값: `500`
- `output_dir`: 체크포인트를 저장하는 위치입니다. (기존 코드에서 하위 디렉토리를 하나 추가했고, stage1의 베이스라인 코드에서 경로를 늘리는 코드를 가져와 자동으로 새로운 경로를 만들어 내도록 했습니다. expr3, expr4 ...) 기본값: `./results/expr`
- `save_steps` : 모델의 파라미터 및 설정값, 체크포인트를 저장하는 빈도입니다. 기본값: `500`
- `save_total_limit`: 체크포인트를 저장하는 최대 갯수입니다. 기본값: `3`
- `logging_steps`: 학습시 loss, epoch, learning_rate 등을 띄우는 빈도입니다. 기본값: `16`
- `logging_dir`: tensorboard 로그를 저장하는 경로입니다. 기본값: `./log`

**추론 관련 하이퍼파라미터 (inference.py)**

- `model_dir` : 학습된 모델의 경로입니다. 기본값: `./results/expr/checkpoint-2000`
- `out_path`: 추론 결과를 저장할 csv 파일의 경로입니다. 기본값: `./predidction/submission.csv`

### How to

---

주의사항: `inference.py` 실행하시기 전에 동일한 경로에 prediction 폴더를 만드셔야 합니다!

```bash
python train.py --model_type Electra --pretrained_model monologg/koelectra-base-v3-discriminator

python inference.py --model_dir ./results/checkpoint-2000 --out_path ./prediction/submission_2.csv --model_type Electra --pretrained_model monologg/koelectra-base-v3-discriminator
```



## Inference
---
* python inference.py --model_dir=[model_path]
* ex) python inference.py --model_dir=./results/checkpoint-500

## Ensemble
---
- K-fold 앙상블 학습
    
    ex) 
    ``` bash
    python ensemble_train.py --model_type XLMRoberta --model_task ForSequenceClassification --epochs 10 --output_dir results/xlm-roberta --n_folds 8 --criterion LabelSmoothing --seed 42 --batch_size 32
    ```

 -  앙상블 추론
    - Soft Voting


        soft ensemble할 리스트의 경우 파일에서 직접 수정해서 soft_ensemble_list에 넣어 주어야 합니다.    
        ```
        python inference_ensemble.py --inference_type soft --output_dir ./prediction/ensemble_soft
        ```
    - Hard Voting

        hard ensemble할 리스트의 경우 파일에서 직접 수정해서 soft_ensemble_list에 넣어 주어야 합니다.    

        ```
        python inference_ensemble.py --inference_type soft --output_dir ./prediction/ensemble_soft
        ```
