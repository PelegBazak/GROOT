import os
import json
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split

RANDOM_STATE = 6
DATA_DIR = "/Data/"
INPUT_FILE_NAME = "full_df_with_features.csv"
OUTPUT_DIR = "/Data/train_results/"


def select_top_feature(model, X, y, n_splits=5, step=0.01):
    model_name = type(model.steps[-1][1]).__name__
    min_features_to_select = 1  # Minimum number of features to consider
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rfecv = RFECV(
        estimator=model,
        step=step,
        cv=cv,
        scoring='r2',
        min_features_to_select=min_features_to_select,
        n_jobs=1,
        importance_getter=f'named_steps.{model_name.lower()}.feature_importances_',
    )
    rfecv.fit(X, y)
    top_features = rfecv.feature_names_in_[rfecv.support_]
    cv_results = pd.DataFrame(rfecv.cv_results_)
    return top_features, cv_results


if __name__ == '__main__':
    full_df = pd.read_csv(os.path.join(DATA_DIR, "large_dataset_df.csv"), index_col=0)
    y = full_df.fold_change
    X_train, X_test, y_train, y_test = train_test_split(full_df, y, test_size=0.2, random_state=RANDOM_STATE)

    catboost_top_features, catboost_cv_results = select_top_feature(
        make_pipeline(RobustScaler(), CatBoostRegressor(
            allow_writing_files=False, random_state=RANDOM_STATE, task_type="GPU", verbose=False)), X_train, y_train
    )
    catboost_cv_results.to_csv(os.path.join(OUTPUT_DIR, f'catboost_cv_results.csv'))

    with open(os.path.join(OUTPUT_DIR, f'catboost_top_features.json'), "w") as f:
        f.write(json.dumps(list(catboost_top_features), indent=4))

    lgbm_top_features, lgbm_cv_results = select_top_feature(
        make_pipeline(RobustScaler(), LGBMRegressor(random_state=RANDOM_STATE, device='gpu', verbose=-1)),
        X_train, y_train
    )
    lgbm_cv_results.to_csv(os.path.join(OUTPUT_DIR, f'lgbm_cv_results.csv'))

    with open(os.path.join(OUTPUT_DIR, f'lgbm_top_features.json'), "w") as f:
        f.write(json.dumps(list(lgbm_top_features), indent=4))

    def objective(trial):
        param_distributions = {
            "random_state": RANDOM_STATE,
            "device": "gpu",
            'verbose': -1,
            "boosting_type": trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 16, log=False),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 3, log=False),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 3, log=False),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1, log=False),
            "subsample": trial.suggest_float("subsample", 0.1, 1, log=False),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 1, log=False),
        }
        model = make_pipeline(RobustScaler(), LGBMRegressor(**param_distributions))
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        score = cross_val_score(model, X_train[lgbm_top_features], y_train, n_jobs=1, cv=cv, scoring='r2')
        r2 = score.mean()
        return r2


    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    study_df = study.trials_dataframe(attrs=('number', 'value', 'datetime_start', 'datetime_complete', 'duration',
                                             'params', 'user_attrs', 'system_attrs', 'state'))
    study_df.to_csv(os.path.join(OUTPUT_DIR, f'lgbm_optuna_study_results.csv'))
    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)
    print("Best Trial: ", study.best_trial)

    def objective(trial):
        param_distributions = {
            "random_state": RANDOM_STATE,
            "task_type": "GPU",
            "allow_writing_files": False,
            "silent": True,
            "iterations": trial.suggest_int("iterations", 10, 20000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 100, log=False),
            "depth": trial.suggest_int("depth", 1, 10, log=False),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100, log=True),
        }
        model = make_pipeline(RobustScaler(), CatBoostRegressor(**param_distributions))
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        score = cross_val_score(model, X_train[catboost_cv_results], y_train, n_jobs=1, cv=cv, scoring='r2')
        r2 = score.mean()
        return r2

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    study_df = study.trials_dataframe(attrs=('number', 'value', 'datetime_start', 'datetime_complete', 'duration',
                                             'params', 'user_attrs', 'system_attrs', 'state'))
    study_df.to_csv(os.path.join(OUTPUT_DIR, f'catboost_optuna_study_results.csv'))
    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)
    print("Best Trial: ", study.best_trial)
