import os
import json

import numpy as np
import torch
import pandas as pd
import xgboost as xgb
from torch import nn, optim
import lightning as pl
from lightning.pytorch import seed_everything
import torch.utils.data as data_utils
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr, pearsonr
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import RobustScaler, OneHotEncoder

RANDOM_STATE = 6
INPUT_FILE_PATH = "/Data/X_train_with_fold_change_label.csv"
OUTPUT_DIR = "/Data/train_results"
# top 200 features found using mRMR
FEATURES = []

seed_everything(RANDOM_STATE, workers=True)


class SoftOrdering1DCNN(pl.LightningModule):

    def __init__(self, input_dim, output_dim, sign_size=32, cha_input=16, cha_hidden=32,
                 K=2, dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2):
        super().__init__()

        hidden_size = sign_size * cha_input
        sign_size1 = sign_size
        sign_size2 = sign_size // 2
        output_size = (sign_size // 4) * cha_hidden

        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output

        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = conv1 = nn.Conv1d(
            cha_input,
            cha_input * K,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=cha_input,
            bias=False)
        self.conv1 = nn.utils.weight_norm(conv1, dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input * K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input * K,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)

        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
        conv4 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=cha_hidden,
            bias=False)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)

        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm2 = nn.BatchNorm1d(output_size)
        self.dropout2 = nn.Dropout(dropout_output)
        dense2 = nn.Linear(output_size, output_dim, bias=False)
        self.dense2 = nn.utils.weight_norm(dense2)

        self.loss = nn.L1Loss()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))

        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))

        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x = x + x_s
        x = nn.functional.relu(x)

        x = self.avg_po_c4(x)

        x = self.flt(x)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self.forward(x), y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        return self(x)

    def _prepare_batch(self, batch):
        x, _ = batch
        return x.view(x.size(0), -1)


if __name__ == '__main__':
    gpu_models = [
        (LGBMRegressor, {"random_state": RANDOM_STATE, "device": 'gpu', "verbose": -1}),
        (xgb.XGBRegressor, {
            "random_state": RANDOM_STATE, "objective": "reg:squarederror", "tree_method": "hist",
            "device": "cuda", "verbose": False
        }),
        (CatBoostRegressor, {
            "allow_writing_files": False, "random_state": RANDOM_STATE, "task_type": "GPU", "verbose": False
        }),
        (TabNetRegressor, {"seed": RANDOM_STATE, "device_name": "cuda"})
    ]

    X_train = pd.read_csv(INPUT_FILE_PATH, index_col=0)
    y_train = X_train['fold_change']
    X_train = X_train.drop(columns=['fold_change'])
    X_train = X_train[FEATURES]

    for model_type, params in gpu_models:
        model = make_pipeline(RobustScaler(), model_type(**params))
        model_name = type(model[-1]).__name__
        spearman_results = []
        pearson_results = []
        r_squared_results = []
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for train, test in kf.split(X_train):
            X_fold_train, X_fold_test, y_fold_train, y_fold_test = (
                X_train.iloc[train], X_train.iloc[test], y_train.iloc[train], y_train.iloc[test]
            )
            y_fold_train = y_fold_train.to_numpy().reshape(-1, 1)
            y_fold_test = y_fold_test.to_numpy().reshape(-1, 1)
            model.fit(X_fold_train, y_fold_train)
            predictions = model.predict(X_fold_test).squeeze()
            spearman_results.append(spearmanr(predictions, y_fold_test.squeeze()))
            pearson_results.append(pearsonr(predictions, y_fold_test.squeeze()))
            r_squared_results.append(r2_score(y_fold_test.squeeze(), predictions))

        results = {
            "model_type": model_name,
            "model_params": json.dumps(params),
            "features": FEATURES,
            "spearman_results": spearman_results,
            "pearson_results": pearson_results,
            "r_squared_results": r_squared_results
        }

        with open(os.path.join(OUTPUT_DIR, f'{model_name}.json'), "w") as f:
            f.write(json.dumps(results, indent=4))

    # GBDT+LR
    model_name = "GBDT_LR"

    spearman_results = []
    pearson_results = []
    r_squared_results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for train, test in kf.split(X_train):
        X_fold_train, X_fold_test, y_fold_train, y_fold_test = (
            X_train.iloc[train], X_train.iloc[test], y_train.iloc[train], y_train.iloc[test]
        )

        base_model = make_pipeline(
            RobustScaler(),
            xgb.XGBRegressor(
                random_state=RANDOM_STATE, objective="reg:squarederror", tree_method="hist", device="cuda",
                verbose=False
            )
        )
        head_model = Ridge(random_state=RANDOM_STATE)

        base_model.fit(X_fold_train, y_fold_train)
        booster = base_model[-1].get_booster()
        train_leaf_indices = pd.DataFrame(
            booster.predict(xgb.DMatrix(X_fold_train), pred_leaf=True).reshape(X_fold_train.shape[0], -1))

        encoder = OneHotEncoder(
            categories=booster.trees_to_dataframe().groupby('Tree').agg({'Node': lambda x: x.tolist()}).Node.to_list())
        X_fold_train_binary = encoder.fit_transform(train_leaf_indices)

        head_model.fit(X_fold_train_binary, y_fold_train)

        test_leaf_indices = pd.DataFrame(
            booster.predict(xgb.DMatrix(X_fold_test), pred_leaf=True).reshape(X_fold_test.shape[0], -1))

        X_fold_test_binary = encoder.transform(test_leaf_indices)

        predictions = head_model.predict(X_fold_test_binary)
        spearman_results.append(spearmanr(predictions, y_fold_test))
        pearson_results.append(pearsonr(predictions, y_fold_test))
        r_squared_results.append(r2_score(y_fold_test, predictions))

    results = {
        "model_type": model_name,
        "features": FEATURES,
        "spearman_results": spearman_results,
        "pearson_results": pearson_results,
        "r_squared_results": r_squared_results
    }

    with open(os.path.join(OUTPUT_DIR, f'{model_name}.json'), "w") as f:
        f.write(json.dumps(results, indent=4))

    # 1D CNN
    model_name = "1dCNN"

    spearman_results = []
    pearson_results = []
    r_squared_results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for train, test in kf.split(X_train):
        X_fold_train, X_fold_test, y_fold_train, y_fold_test = (
            X_train.iloc[train], X_train.iloc[test], y_train.iloc[train], y_train.iloc[test]
        )

        y_tensor = torch.tensor(y_fold_train.values.astype(np.float32))
        X_tensor = torch.tensor(X_fold_train.values.astype(np.float32))
        train_tensor = data_utils.TensorDataset(X_tensor, y_tensor)
        train_loader = data_utils.DataLoader(dataset=train_tensor, batch_size=32, shuffle=True)
        trainer = pl.Trainer(max_epochs=35, accelerator="gpu")
        model = SoftOrdering1DCNN(200, 1)
        trainer.fit(model=model, train_dataloaders=train_loader)
        y_test_tensor = torch.tensor(y_fold_test.values.astype(np.float32))
        X_test_tensor = torch.tensor(X_fold_test.values.astype(np.float32))
        val_tensor = data_utils.TensorDataset(X_test_tensor, y_test_tensor)
        val_loader = data_utils.DataLoader(dataset=val_tensor, batch_size=32, shuffle=False)
        predictions = trainer.predict(model, val_loader)
        predictions_flat = [
            x
            for xs in predictions
            for x in xs
        ]
        spearman_results.append(spearmanr(predictions_flat, y_fold_test))
        pearson_results.append(pearsonr(predictions_flat, y_fold_test))
        r_squared_results.append(r2_score(y_fold_test, predictions_flat))

    results = {
        "model_type": model_name,
        "features": FEATURES,
        "spearman_results": spearman_results,
        "pearson_results": pearson_results,
        "r_squared_results": r_squared_results
    }

    with open(os.path.join(OUTPUT_DIR, f'{model_name}.json'), "w") as f:
        f.write(json.dumps(results, indent=4))

