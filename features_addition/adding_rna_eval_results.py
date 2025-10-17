import os
import re
import subprocess

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client
from dask_jobqueue import PBSCluster

DATA_DIR = "/Data/"
WORKING_DIRECTORY = ""
RNAEVAL_BINARY_NAME = "/bin/RNAeval"
POSSIBLE_ENERGY_CONTRIBUTIONS = list({-3.4, -3.3, -2.4, -2.2, -2.1, -1.5, -1.4, -1.3, -1.1, -0.9, -0.6, 0.2, 0.4, 0.9,
                                      1.0, 1.1, 1.2, 1.6, 1.7, 1.8, 1.9, 2.0, 2.4, 2.6, 2.7, 3.4})

def run_rna_eval(sequence, structure):
    return subprocess.check_output(
        [RNAEVAL_BINARY_NAME, "-v", "-d2"],
        universal_newlines=True,
        input=f"{sequence}\n{structure}",
        text=True
    )


def vectorize_contributions(contributions):
    vector = np.array([0 for i in range(len(POSSIBLE_ENERGY_CONTRIBUTIONS))], dtype='float')

    for contribution in contributions:
        if contribution not in POSSIBLE_ENERGY_CONTRIBUTIONS:
            print(contribution)
            continue

        i = POSSIBLE_ENERGY_CONTRIBUTIONS.index(contribution)
        vector[i] += 1

    vector /= len(contributions)
    return vector


def get_basic_stats(rna_eval_result: str):
    external_loop = 0
    hairpin_loop = 0
    interior_loop = []

    for line in rna_eval_result.split('\n'):
        try:
            regex_result = re.search(":\s*[0-9-]+", line).group()
            value = int(regex_result.replace(':', '').strip())
            value /= 100
        except AttributeError:
            break

        if 'External loop' in line:
            external_loop = value
        elif 'Interior loop' in line:
            interior_loop.append(value)
        else:
            hairpin_loop = value

    return (external_loop, hairpin_loop, np.mean(interior_loop), np.std(interior_loop),
            np.max(interior_loop), np.min(interior_loop), vectorize_contributions(interior_loop))


if __name__ == '__main__':
    all_triggers_df = pd.read_csv(os.path.join(DATA_DIR, "full_df.csv"), index_col=0)

    cluster = PBSCluster(
        cores=1,
        memory="1GB",
        queue="",
        walltime="24:00:00",
        scheduler_options={"dashboard_address": ":12435"},
        log_directory="dask-logs",
        job_script_prologue=[f"cd {WORKING_DIRECTORY}"],
    )
    cluster.adapt(minimum=1, maximum=10)
    client = Client(cluster)
    triggers_ddf = dd.from_pandas(all_triggers_df, npartitions=1000)
    triggers_ddf["RNAeval_result"] = triggers_ddf.apply(
        lambda row: run_rna_eval(
            row["switch"][:len(row["ideal_stem_structure"])], row["ideal_stem_structure"]
        ), axis=1, meta=str
    )

    triggers_ddf[[
        "stem_external_loop_contribution", "stem_hairpin_contribution", "stem_mean_energy_contribution",
        "stem_std_energy_contribution", "stem_max_energy_contribution", "stem_min_energy_contribution",
        "interior_loop_vectorized"
    ]] = triggers_ddf.apply(
        lambda row: get_basic_stats(row["RNAeval_result"]),
        axis=1, result_type='expand',
        meta={0: 'float64', 1: 'float64', 2: 'float64', 3: 'float64', 4: 'float64', 5: 'float64', 6: 'object'})

    all_triggers_df = triggers_ddf.compute()
    all_triggers_df.to_csv(os.path.join(DATA_DIR, "full_df_with_rnaeval.csv"))
