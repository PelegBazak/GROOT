import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = "/Data/"
RISEARCH_BINARY_NAME = "/bin/RIsearch1"


def get_partition_function(switch: str) -> float:
    with open("/EternaData/EternaFold/switch.seq", "w") as f:
        f.write(f"{switch}\n")

    try:
        result = subprocess.check_output(["module load openmpi/openmpi-5.0.6-HPC_SDK; module load gcc/rocky8-gcc-13.1.0; /EternaData/EternaFold/src/contrafold predict /EternaData/EternaFold/switch.seq --params /EternaData/EternaFold/parameters/EternaFoldParams.v1 --partition"],
            universal_newlines=True,
            text=True,
            shell=True
        )
    except subprocess.CalledProcessError as e:
        print(e)
        print(switch)
        raise

    return float(re.findall("[0-9.]+", result)[-1])


def get_bppm_features(switch: str):
    with open("/EternaGame/EternaFold/switch.seq", "w") as f:
        f.write(f"{switch}\n")

    subprocess.check_output(
        [
            "module load openmpi/openmpi-5.0.6-HPC_SDK; module load gcc/rocky8-gcc-13.1.0; /EternaGame/EternaFold/src/contrafold predict /EternaGame/EternaFold/switch.seq --params /EternaGame/EternaFold/parameters/EternaFoldParams.v1 --posteriors 0.00001 switch_bps.txt"],
        universal_newlines=True,
        text=True,
        shell=True
    )
    result = Path("./switch_bps.txt").read_text()
    n = len(switch)
    matrix = []

    for line in result.split("\n"):
        row = np.zeros(n)
        results = re.findall("[1-9]+:[e0-9-.]+", line)

        for result in results:
            splitted_result = result.split(":")
            index = int(splitted_result[0]) - 1
            probability = float(splitted_result[1])
            row[index] = probability

        matrix.append(row)

    matrix = np.array(matrix)
    return (
        matrix.tostring(), matrix.mean(), matrix.std(), matrix.max(), matrix.sum(),

        matrix.max(axis=0).mean(), matrix.max(axis=0).std(), matrix.max(axis=0).sum(),
        matrix.mean(axis=0).max(), matrix.mean(axis=0).std(), matrix.mean(axis=0).sum(),
        matrix.sum(axis=0).max(), matrix.sum(axis=0).std(), matrix.sum(axis=0).mean(),
        matrix.std(axis=0).max(), matrix.std(axis=0).sum(), matrix.std(axis=0).mean(),

        matrix.max(axis=1).mean(), matrix.max(axis=1).std(), matrix.max(axis=1).sum(),
        matrix.mean(axis=1).max(), matrix.mean(axis=1).std(), matrix.mean(axis=1).sum(),
        matrix.sum(axis=1).max(), matrix.sum(axis=1).std(), matrix.sum(axis=1).mean(),
        matrix.std(axis=1).max(), matrix.std(axis=1).sum(), matrix.std(axis=1).mean(),

        np.count_nonzero(matrix), np.count_nonzero(matrix, axis=0).mean(), np.count_nonzero(matrix, axis=0).std(),
        np.count_nonzero(matrix, axis=0).max(), np.count_nonzero(matrix, axis=0).sum(),

        np.count_nonzero(matrix, axis=1).mean(), np.count_nonzero(matrix, axis=1).std(),
        np.count_nonzero(matrix, axis=1).max(), np.count_nonzero(matrix, axis=1).sum()
    )


if __name__ == '__main__':
    all_triggers_df = pd.read_csv(
        os.path.join(DATA_DIR, "full_df.csv"),
        index_col=0
    )

    all_triggers_df["eterna_fold_log_partition_coefficient"] = all_triggers_df.switch.apply(get_partition_function)
    all_triggers_df[[
        "eterna_fold_bppm", "eterna_fold_bppm_mean", "eterna_fold_bppm_std", "eterna_fold_bppm_max",
        "eterna_fold_bppm_sum",
        "eterna_fold_bppm_max_mean_0", "eterna_fold_bppm_max_std_0", "eterna_fold_bppm_max_sum_0",
        "eterna_fold_bppm_mean_max_0", "eterna_fold_bppm_mean_std_0", "eterna_fold_bppm_mean_sum_0",
        "eterna_fold_bppm_sum_max_0", "eterna_fold_bppm_sum_std_0", "eterna_fold_bppm_sum_mean_0",
        "eterna_fold_bppm_std_max_0", "eterna_fold_bppm_std_sum_0", "eterna_fold_bppm_std_mean_0",
        "eterna_fold_bppm_max_mean_1", "eterna_fold_bppm_max_std_1", "eterna_fold_bppm_max_sum_1",
        "eterna_fold_bppm_mean_max_1", "eterna_fold_bppm_mean_std_1", "eterna_fold_bppm_mean_sum_1",
        "eterna_fold_bppm_sum_max_1", "eterna_fold_bppm_sum_std_1", "eterna_fold_bppm_sum_mean_1",
        "eterna_fold_bppm_std_max_1", "eterna_fold_bppm_std_sum_1", "eterna_fold_bppm_std_mean_1",
        "eterna_fold_bppm_count_nonzero",
        "eterna_fold_bppm_count_nonzero_mean_0", "eterna_fold_bppm_count_nonzero_std_0",
        "eterna_fold_bppm_count_nonzero_max_0", "eterna_fold_bppm_count_nonzero_sum_0",
        "eterna_fold_bppm_count_nonzero_mean_1", "eterna_fold_bppm_count_nonzero_std_1",
        "eterna_fold_bppm_count_nonzero_max_1", "eterna_fold_bppm_count_nonzero_sum_1",
    ]] = all_triggers_df.apply(lambda row: get_bppm_features(row.switch), axis=1, result_type="expand")

    all_triggers_df.to_csv(os.path.join(DATA_DIR, "full_df_with_eterna_fold_features.csv"))
