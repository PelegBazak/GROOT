import RNA
import time
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import exponnorm, gamma, lognorm, expon, FitError
from dask.distributed import Client
from dask_jobqueue import PBSCluster

POTENTIAL_TRIGGERS_FILE_PATH = "/Data/full_df.csv"
OUTPUT_FILE_PATH = "/Data/full_df_with_distribution_fitting_features.csv"
SAMPLE_SIZE = 10000


def sample_energy_distribution(sequence: str, sample_size: int) -> np.array:
    md = RNA.md()
    md.noLP = 1  # 20 Deg Celcius
    md.dangles = 2  # Dangle Model 1
    md.uniq_ML = 1

    fc = RNA.fold_compound(sequence, md)
    (pp, pf) = fc.pf()
    sampled_structures = fc.pbacktrack(sample_size)
    energies = np.array([fc.eval_structure(sampled_structure) for sampled_structure in sampled_structures])
    return energies


def get_distribution_fitting_features(sequence: str):
    energies = sample_energy_distribution(sequence, SAMPLE_SIZE)
    dist_to_features = {}

    for dist in exponnorm, gamma, lognorm, expon:
        try:
            dist_args = dist.fit(energies)
            log_likelihood = np.sum(np.log(dist.pdf(energies, *dist_args)))
            dist_to_features[dist.name] = (dist_args, log_likelihood)
        except FitError:
            dist_to_features[dist.name] = (None, None)

    return dist_to_features


if __name__ == '__main__':
    working_directory = ""
    cluster = PBSCluster(
        cores=1,
        memory="1GB",
        queue="",
        walltime="48:00:00",
        scheduler_options={"dashboard_address": ":12435"},
        log_directory="dask-logs",
        job_script_prologue=[f"cd {working_directory}"],
    )
    cluster.adapt(minimum=1, maximum=65)
    client = Client(cluster)
    triggers_df = pd.read_csv(POTENTIAL_TRIGGERS_FILE_PATH, index_col=0)
    triggers_ddf = dd.from_pandas(triggers_df, npartitions=65 * 10)

    triggers_ddf["switch_distribution_fitting_features"] = triggers_ddf.apply(
        lambda row: get_distribution_fitting_features(row["switch"]), axis=1, meta=dict)

    triggers_ddf["stem_distribution_fitting_features"] = triggers_ddf.apply(
        lambda row: get_distribution_fitting_features(
            row["switch"][int(row["stem_start"]):int(row["stem_end"]) + 1]
        ), axis=1, meta=dict)

    triggers_ddf["loop_to_stem_end_distribution_fitting_features"] = triggers_ddf.apply(
        lambda row: get_distribution_fitting_features(
            row["switch"][int(row["loop_start"]):int(row["stem_end"]) + 1]
        ), axis=1, meta=dict)

    triggers_ddf["loop_to_end_distribution_fitting_features"] = triggers_ddf.apply(
        lambda row: get_distribution_fitting_features(
            row["switch"][int(row["loop_start"]):],
        ), axis=1, meta=dict)

    triggers_ddf["stem_start_to_loop_end_distribution_fitting_features"] = triggers_ddf.apply(
        lambda row: get_distribution_fitting_features(
            row["switch"][int(row["stem_start"]):int(row["loop_end"]) + 1],
        ), axis=1, meta=dict)

    triggers_ddf["start_to_loop_end_distribution_fitting_features"] = triggers_ddf.apply(
        lambda row: get_distribution_fitting_features(
            row["switch"][:int(row["loop_end"]) + 1],
        ), axis=1, meta=dict)

    triggers_ddf["stem_top_distribution_fitting_features"] = triggers_ddf.apply(
        lambda row: get_distribution_fitting_features(
            row["switch"][int(row["stem_top_start"]):int(row["stem_top_end"]) + 1],
        ), axis=1, meta=dict)

    triggers_df = triggers_ddf.compute()
    triggers_df.to_csv(OUTPUT_FILE_PATH)
