from mrmr import mrmr_regression
import pandas as pd

INPUT_FILE_PATH = "/Data/X_train_with_fold_change_label.csv"
LABEL_COLUMN = 'fold_change'


if __name__ == '__main__':
    X_train = pd.read_csv(INPUT_FILE_PATH, index_col=0)
    y_train = X_train[LABEL_COLUMN]
    X_train = X_train.drop(columns=[LABEL_COLUMN])

    selected_features = mrmr_regression(X=X_train, y=y_train, K=200)
    print(selected_features)
