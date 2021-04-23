#python inference.py --model_dir ./results/ensemble_final7/model --out_path ./prediction/ensemble_final7/model1.csv --model_type Electra --pretrained_model monologg/koelectra-base-v3-discriminator --inference_type logits

#python inference.py --model_dir ./results/ensemble_final7/model2 --out_path ./prediction/ensemble_final7/model2.csv --model_type Electra --pretrained_model monologg/koelectra-base-v3-discriminator --inference_type logits

#python inference.py --model_dir ./results/ensemble_final7/model2 --out_path ./prediction/ensemble_final7/model2.csv --model_type Electra --pretrained_model monologg/koelectra-base-v3-discriminator --inference_type labels

#python inference.py --model_dir ./results/ensemble_final7/model2 --out_path ./prediction/ensemble_final7/model2.csv --model_type Electra --pretrained_model monologg/koelectra-base-v3-discriminator --inference_type logits

#python inference.py --model_dir ./results/ensemble_final7/model3 --out_path ./prediction/ensemble_final7/model3.csv --model_type Electra --pretrained_model monologg/koelectra-base-v3-discriminator --inference_type logits

#python inference.py --model_dir ./results/ensemble_final7/model4/checkpoint-500 --out_path ./prediction/ensemble_final7/model4.csv --model_type Electra --pretrained_model monologg/koelectra-base-v3-discriminator --inference_type logits


python inference.py --model_dir ./results/ensemble_final7/model --out_path ./prediction/ensemble_final7_hard/model1.csv --model_type Electra --pretrained_model monologg/koelectra-base-v3-discriminator --inference_type labels 


python inference.py --model_dir ./results/ensemble_final7/model2 --out_path ./prediction/ensemble_final7_hard/model2.csv --model_type Electra --pretrained_model monologg/koelectra-base-v3-discriminator --inference_type labels

python inference.py --model_dir ./results/ensemble_final7/model3 --out_path ./prediction/ensemble_final7_hard/model3.csv --model_type Electra --pretrained_model monologg/koelectra-base-v3-discriminator --inference_type labels

python inference.py --model_dir ./results/ensemble_final7/model4/checkpoint-500 --out_path ./prediction/ensemble_final7_hard/model4.csv --model_type Electra --pretrained_model monologg/koelectra-base-v3-discriminator --inference_type labels
