import numpy as np
import pandas as pd
import argparse

from scipy.stats import mode


def soft_ensemble(ensemble_list, output_dir):
    ensemble_logits = np.zeros((1000, 42))

    for i in range(len(ensemble_list)):
        logits = pd.read_csv(ensemble_list[i])
        logits = logits.to_numpy()
        ensemble_logits += logits

    print(ensemble_logits.shape)
    print(ensemble_logits[0])

    result = np.argmax(ensemble_logits, axis=1)
    output_pred = list(result)
    len(output_pred)
    pred_answer =  list(np.array(output_pred).reshape(-1))

    # make csv file with predicted answer
    output = pd.DataFrame(pred_answer, columns=['pred'])
    output.to_csv(f'{output_dir}/soft.csv', index=False)


# ## Hard
def hard_ensemble(ensemble_list, output_dir):
    output_pred = []
    for i in range(len(ensemble_list)):
        pred = pd.read_csv(ensemble_list[i])
        result = np.array(pred['pred'].tolist())
        output_pred.append(list(np.array(result).reshape(-1)))

    from scipy.stats import mode

    ensemble_output = []
    for i in range(1000):
        ensemble_output.append(mode([output_pred[0][i], output_pred[1][i], output_pred[2][i]])[0][0])

    # make csv file with predicted answer
    output = pd.DataFrame(ensemble_output, columns=['pred'])
    output.to_csv(f'{output_dir}/hard.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voting_type", type=str, default="soft")
    parser.add_argument("--output_dir", type=str, default="./prediction/ensemble")
    args = parser.parse_args()

    soft_ensemble_list = []
    hard_ensemble_list = ["/opt/ml/code/prediction/submission_corrected_data.csv", "/opt/ml/code/prediction/submission_new.csv", "/opt/ml/code/prediction/submission.csv", "/opt/ml/code/prediction/good_results/output (1).csv", "/opt/ml/code/prediction/good_results/output (2).csv", "/opt/ml/code/prediction/good_results/output.csv", "/opt/ml/code/prediction/soft-koelectra-8fold.csv"]

    # soft_ensemble_list = ["/opt/ml/code/prediction/fold-xlm-roberta1-submission.csv", "/opt/ml/code/prediction/koelectra-expr15-submission.csv", "/opt/ml/code/prediction/xlm-roberta1-submission.csv"] #7
    # soft_ensemble_list = ["/opt/ml/code/prediction/ensemble_final7/model1.csv", "/opt/ml/code/prediction/ensemble_final7/model2.csv", "/opt/ml/code/prediction/ensemble_final7/model3.csv"] #9
    # hard_ensemble_list = ["/opt/ml/code/prediction/submission_corrected_data.csv", "/opt/ml/code/prediction/submission_new.csv", "/opt/ml/code/prediction/submission.csv", "/opt/ml/code/prediction/good_results/output (1).csv", "/opt/ml/code/prediction/good_results/output (2).csv", "/opt/ml/code/prediction/good_results/output.csv"] #8

    if args.voting_type == "soft":
        soft_ensemble(soft_ensemble_list, args.output_dir)
    else:
        hard_ensemble(hard_ensemble_list, args.output_dir)

