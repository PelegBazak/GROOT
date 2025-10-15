import os
import sys
sys.path.insert(1, '..')
from utils.pbs_job_submission import submit_job

RANDOM_STATE = 6
INPUT_FILE_PATH = "/Data/X_train_with_fold_change_label.csv"
OUTPUT_DIR = "/Data/train_results"
# top 200 features found using mRMR
FEATURES = ""

if __name__ == '__main__':
    model_to_params = {
        "linear_model.Ridge": f"{{\\\"random_state\\\":{RANDOM_STATE}}}",
        "linear_model.Lasso": f"{{\\\"random_state\\\":{RANDOM_STATE}}}",
        "linear_model.ElasticNet": f"{{\\\"random_state\\\":{RANDOM_STATE}}}",
        "svm.SVR": f"{{}}",
        "ensemble.AdaBoostRegressor": f"{{\\\"random_state\\\":{RANDOM_STATE}}}",
        "ensemble.ExtraTreesRegressor": f"{{\\\"random_state\\\":{RANDOM_STATE}}}",
        "ensemble.BaggingRegressor": f"{{\\\"random_state\\\":{RANDOM_STATE}}}",
        "neural_network.MLPRegressor": f"'{{\"random_state\":{RANDOM_STATE}, \"hidden_layer_sizes\":[100, 50]}}'"
    }

    for model_type, params in model_to_params.items():
        print(params)
        submit_job(
            model_type,
            f"python /GROOT/models_training/train_sklearn_model.py"
            f" --input_file_path {INPUT_FILE_PATH}"
            f" --output_file_path {os.path.join(OUTPUT_DIR, model_type.split('.')[-1] + '.json')}"
            f" --model_type {model_type}"
            f" --model_params {params}"
            f" --features {FEATURES}",
            ram=20, hours_to_run=24, cores_count=1,
            queue_name=""
        )
