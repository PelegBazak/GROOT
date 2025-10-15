import json
import argparse
import importlib
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, pearsonr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 6
LABEL_COLUMN = 'fold_change'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, help="the path to the input file")
    parser.add_argument("--output_file_path", type=str, help="the path to the output file")
    parser.add_argument("--model_type", type=str, help="type of model to train")
    parser.add_argument("--model_params", type=str, help="params of the model to train")
    parser.add_argument("--features", nargs="*", type=str, help="list of features to use")
    args = parser.parse_args()

    X_train = pd.read_csv(args.input_file_path, index_col=0)
    y_train = X_train[LABEL_COLUMN]
    X_train = X_train.drop(columns=[LABEL_COLUMN])

    if args.features:
        X_train = X_train[args.features]

    module_name, model_name = args.model_type.split(".")
    model_type = getattr(importlib.import_module(f"sklearn.{module_name}"), model_name)
    model = make_pipeline(
        RobustScaler(),
        model_type(**json.loads(args.model_params))
    )

    kf = KFold(n_splits=5, shuffle=True)
    spearman_results = []
    pearson_results = []
    r_squared_results = []

    for train, test in kf.split(X_train):
        X_fold_train, X_fold_test, y_fold_train, y_fold_test = (
            X_train.iloc[train], X_train.iloc[test], y_train.iloc[train], y_train.iloc[test]
        )
        model.fit(X_fold_train, y_fold_train)
        spearman_results.append(spearmanr(model.predict(X_fold_test), y_fold_test))
        pearson_results.append(pearsonr(model.predict(X_fold_test), y_fold_test))
        r_squared_results.append(r2_score(y_fold_test, model.predict(X_fold_test)))

    results = {
        "model_type": args.model_type,
        "model_params": args.model_params,
        "features": list(X_train.columns),
        "spearman_results": spearman_results,
        "pearson_results": pearson_results,
        "r_squared_results": r_squared_results
    }

    with open(args.output_file_path, "w") as f:
        f.write(json.dumps(results, indent=4))
