import json
import os
import re
import subprocess

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client
from dask_jobqueue import PBSCluster

INPUT_FILE = "full_df.csv"
OUTPUT_FILE = "full_df_with_rrna_mrna_interaction_strength_profiles.csv"
DATA_DIR = "/Data/"
RNACOFOLD_BINARY_NAME = "/bin/RNAcofold"
WORKING_DIRECTORY = ""
ANTI_SHINE_DELGARNO = "UCCUCC"


def run_rna_cofold(sequence1, sequence2):
    try:
        return subprocess.check_output(
            [RNACOFOLD_BINARY_NAME, "-d2", "--noLP"],
            universal_newlines=True,
            input=f"{sequence1}&{sequence2}",
            text=True
        )
    except Exception as e:
        print(f"Error while running RNAcofold on {sequence1} and {sequence2}. Error: {e}")
        raise


def get_mfe(switch) -> float:
    rna_cofold_output = run_rna_cofold(ANTI_SHINE_DELGARNO, switch)

    try:
        regex_result = re.search("\(\s*[0-9-]+[0-9.]*\)", rna_cofold_output).group()
    except Exception as e:
        print(e)
        print(switch)
        return 0

    mfe = float(regex_result.replace('(', '').replace(')', ''))
    return mfe


def get_mfe_profile(switch) -> tuple[str, float, float, float, float]:
    mfe_profile = []

    for i in range(0, len(switch) - len(ANTI_SHINE_DELGARNO) + 1):
        mfe_profile.append(get_mfe(switch[i:i+len(ANTI_SHINE_DELGARNO)]))

    return json.dumps(mfe_profile), min(mfe_profile), max(mfe_profile), np.mean(mfe_profile), np.std(mfe_profile)


if __name__ == '__main__':
    triggers_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE))[["switch", "trigger", "origin"]]
    cluster = PBSCluster(
        cores=1,
        memory="5GB",
        queue="",
        walltime="72:00:00",
        scheduler_options={"dashboard_address": ":12435"},
        log_directory="dask-logs",
        job_script_prologue=[f"cd {WORKING_DIRECTORY}"],
    )
    cluster.adapt(minimum=1, maximum=10)
    client = Client(cluster)
    triggers_ddf = dd.from_pandas(triggers_df, npartitions=1000)
    triggers_ddf[[
        "rrna_mrna_interaction_strength_profile", 'rrna_mrna_interaction_strength_profile_min',
        'rrna_mrna_interaction_strength_profile_max', 'rrna_mrna_interaction_strength_profile_mean',
        'rrna_mrna_interaction_strength_profile_std']] = triggers_ddf.apply(
        lambda row: get_mfe_profile(row.switch), axis=1, result_type='expand',
        meta={0: 'str', 1: 'float64', 2: 'float64', 3: 'float64', 4: 'float64'}
    )
    triggers_df = triggers_ddf.compute()
    triggers_df.to_csv(os.path.join(DATA_DIR, OUTPUT_FILE))
