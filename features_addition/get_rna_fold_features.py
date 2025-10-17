import RNA
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client
from dask_jobqueue import PBSCluster

POTENTIAL_TRIGGERS_FILE_PATH = "/Data/full_df.csv"
OUTPUT_FILE_PATH = "/Data/full_df_with_rna_fold_features.csv"
SAMPLE_SIZE = 10000


def get_energy_properties(sequence: str, ideal_stem_structure: str = None):
    # create a new model details structure
    md = RNA.md()

    # change temperature and dangle model
    md.noLP = 1  # 20 Deg Celcius
    md.dangles = 2  # Dangle Model 1
    md.uniq_ML = 1

    fc = RNA.fold_compound(sequence, md)
    (mfe_structure, mfe) = fc.mfe()
    fc.exp_params_rescale(mfe)
    (pp, pf) = fc.pf()
    (centroid_structure, distance) = fc.centroid()
    centroid_energy = fc.eval_structure(centroid_structure)
    centroid_ensemble_defect = fc.ensemble_defect(centroid_structure)
    mfe_ensemble_defect = fc.ensemble_defect(mfe_structure)
    centroid_probability = fc.pr_structure(centroid_structure)
    mfe_probability = fc.pr_structure(mfe_structure)
    mfe_match_count = mfe_structure.count("(")
    centroid_match_count = centroid_structure.count("(")

    sampled_structures = fc.pbacktrack(SAMPLE_SIZE)
    energies = np.array([fc.eval_structure(sampled_structure) for sampled_structure in sampled_structures])

    if ideal_stem_structure:
        estimated_probability_for_ideal_stem = sum(
            [ideal_stem_structure == sampled_structure[:len(ideal_stem_structure)]
             for sampled_structure in sampled_structures]) / SAMPLE_SIZE

        return (mfe_structure, mfe, mfe_match_count, mfe_ensemble_defect, mfe_probability,
                centroid_structure, centroid_energy, centroid_match_count, centroid_ensemble_defect,
                centroid_probability,
                pf, estimated_probability_for_ideal_stem, energies.mean(), energies.std())

    return (mfe_structure, mfe, mfe_match_count, mfe_ensemble_defect, mfe_probability,
            centroid_structure, centroid_energy, centroid_match_count, centroid_ensemble_defect,
            centroid_probability,
            pf, energies.mean(), energies.std())


if __name__ == '__main__':
    working_directory = ""
    cluster = PBSCluster(
        cores=1,
        memory="1GB",
        queue="",
        walltime="24:00:00",
        scheduler_options={"dashboard_address": ":12435"},
        log_directory="dask-logs",
        job_script_prologue=[f"cd {working_directory}"],
    )
    cluster.adapt(minimum=1, maximum=10)
    client = Client(cluster)
    triggers_df = pd.read_csv(POTENTIAL_TRIGGERS_FILE_PATH, index_col=0)
    triggers_ddf = dd.from_pandas(triggers_df, npartitions=1000)
    triggers_ddf[[
        "switch_mfe_structure", "switch_mfe", "switch_mfe_matches_count", "switch_mfe_ensemble_defect",
        "switch_mfe_probability", "switch_centroid_structure", "switch_centroid_energy",
        "switch_centroid_matches_count", "switch_centroid_ensemble_defect", "switch_centroid_probability",
        "switch_partition_function",
        "estimated_probability_for_ideal_stem", "switch_estimated_energy_mean", "switch_estimated_energy_std"
    ]] = triggers_ddf.apply(
        lambda row: get_energy_properties(row["switch"], row["ideal_stem_structure"]),
        axis=1, result_type='expand',
        meta={0: 'str', 1: 'float64', 2: 'int64', 3: 'float64', 4: 'float64',
              5: 'str', 6: 'float64', 7: 'int64', 8: 'float64', 9: 'float64', 10: 'float64',
              11: 'float64', 12: 'float64', 13: 'float64'})

    triggers_ddf[[
        "binding_site_mfe_structure", "binding_site_mfe", "binding_site_mfe_matches_count",
        "binding_site_mfe_ensemble_defect", "binding_site_mfe_probability",
        "binding_site_centroid_structure", "binding_site_centroid_energy", "binding_site_centroid_matches_count",
        "binding_site_centroid_ensemble_defect", "binding_site_centroid_probability",
        "binding_site_partition_function",
        "binding_site_estimated_energy_mean", "binding_site_estimated_energy_std"
    ]] = triggers_ddf.apply(
        lambda row: get_energy_properties(
            row["switch"][int(row["trigger_binding_site_start"]):int(row["trigger_binding_site_end"]) + 1]),
        axis=1, result_type='expand',
        meta={0: 'str', 1: 'float64', 2: 'int64', 3: 'float64', 4: 'float64',
              5: 'str', 6: 'float64', 7: 'int64', 8: 'float64', 9: 'float64', 10: 'float64',
              11: 'float64', 12: 'float64'})

    triggers_ddf[[
        "stem_mfe_structure", "stem_mfe", "stem_mfe_matches_count",
        "stem_mfe_ensemble_defect", "stem_mfe_probability",
        "stem_centroid_structure", "stem_centroid_energy", "stem_centroid_matches_count",
        "stem_centroid_ensemble_defect", "stem_centroid_probability", "stem_partition_function",
        "stem_estimated_energy_mean", "stem_estimated_energy_std"
    ]] = triggers_ddf.apply(
        lambda row: get_energy_properties(
            row["switch"][int(row["stem_start"]):int(row["stem_end"]) + 1]),
        axis=1, result_type='expand',
        meta={0: 'str', 1: 'float64', 2: 'int64', 3: 'float64', 4: 'float64',
              5: 'str', 6: 'float64', 7: 'int64', 8: 'float64', 9: 'float64', 10: 'float64',
              11: 'float64', 12: 'float64'})

    triggers_ddf[[
        "loop_mfe_structure", "loop_mfe", "loop_mfe_matches_count",
        "loop_mfe_ensemble_defect", "loop_mfe_probability",
        "loop_centroid_structure", "loop_centroid_energy", "loop_centroid_matches_count",
        "loop_centroid_ensemble_defect", "loop_centroid_probability", "loop_partition_function",
        "loop_estimated_energy_mean", "loop_estimated_energy_std"
    ]] = triggers_ddf.apply(
        lambda row: get_energy_properties(
            row["switch"][int(row["loop_start"]):int(row["loop_end"]) + 1]),
        axis=1, result_type='expand',
        meta={0: 'str', 1: 'float64', 2: 'int64', 3: 'float64', 4: 'float64',
              5: 'str', 6: 'float64', 7: 'int64', 8: 'float64', 9: 'float64', 10: 'float64',
              11: 'float64', 12: 'float64'})

    triggers_ddf[[
        "loop_to_end_mfe_structure", "loop_to_end_mfe", "loop_to_end_mfe_matches_count",
        "loop_to_end_mfe_ensemble_defect", "loop_to_end_mfe_probability",
        "loop_to_end_centroid_structure", "loop_to_end_centroid_energy", "loop_to_end_centroid_matches_count",
        "loop_to_end_centroid_ensemble_defect", "loop_to_end_centroid_probability",
        "loop_to_end_partition_function", "loop_to_end_estimated_energy_mean",
        "loop_to_end_estimated_energy_std"
    ]] = triggers_ddf.apply(
        lambda row: get_energy_properties(
            row["switch"][int(row["loop_start"]):int(row["stem_end"]) + 1]),
        axis=1, result_type='expand',
        meta={0: 'str', 1: 'float64', 2: 'int64', 3: 'float64', 4: 'float64',
              5: 'str', 6: 'float64', 7: 'int64', 8: 'float64', 9: 'float64', 10: 'float64',
              11: 'float64', 12: 'float64'})

    triggers_ddf[[
        "loop_to_stem_end_mfe_structure", "loop_to_stem_end_mfe", "loop_to_stem_end_mfe_matches_count",
        "loop_to_stem_end_mfe_ensemble_defect", "loop_to_stem_end_mfe_probability",
        "loop_to_stem_end_centroid_structure", "loop_to_stem_end_centroid_energy",
        "loop_to_stem_end_centroid_matches_count",
        "loop_to_stem_end_centroid_ensemble_defect", "loop_to_stem_end_centroid_probability",
        "loop_to_stem_end_partition_function", "loop_to_stem_end_estimated_energy_mean",
        "loop_to_stem_end_estimated_energy_std"
    ]] = triggers_ddf.apply(
        lambda row: get_energy_properties(
            row["switch"][int(row["loop_start"]):]),
        axis=1, result_type='expand',
        meta={0: 'str', 1: 'float64', 2: 'int64', 3: 'float64', 4: 'float64',
              5: 'str', 6: 'float64', 7: 'int64', 8: 'float64', 9: 'float64', 10: 'float64',
              11: 'float64', 12: 'float64'})

    triggers_ddf[[
        "stem_start_loop_end_mfe_structure", "stem_start_loop_end_mfe", "stem_start_loop_end_mfe_matches_count",
        "stem_start_loop_end_mfe_ensemble_defect", "stem_start_loop_end_mfe_probability",
        "stem_start_loop_end_centroid_structure", "stem_start_loop_end_centroid_energy",
        "stem_start_loop_end_centroid_matches_count",
        "stem_start_loop_end_centroid_ensemble_defect", "stem_start_loop_end_centroid_probability",
        "stem_start_loop_end_partition_function", "stem_start_loop_end_estimated_energy_mean",
        "stem_start_loop_end_estimated_energy_std"
    ]] = triggers_ddf.apply(
        lambda row: get_energy_properties(
            row["switch"][int(row["stem_start"]):int(row["loop_end"]) + 1]),
        axis=1, result_type='expand',
        meta={0: 'str', 1: 'float64', 2: 'int64', 3: 'float64', 4: 'float64',
              5: 'str', 6: 'float64', 7: 'int64', 8: 'float64', 9: 'float64', 10: 'float64',
              11: 'float64', 12: 'float64'})

    triggers_ddf[[
        "start_loop_end_mfe_structure", "start_loop_end_mfe", "start_loop_end_mfe_matches_count",
        "start_loop_end_mfe_ensemble_defect", "start_loop_end_mfe_probability",
        "start_loop_end_centroid_structure", "start_loop_end_centroid_energy",
        "start_loop_end_centroid_matches_count",
        "start_loop_end_centroid_ensemble_defect", "start_loop_end_centroid_probability",
        "start_loop_end_partition_function", "start_loop_end_estimated_energy_mean",
        "start_loop_end_estimated_energy_std"
    ]] = triggers_ddf.apply(
        lambda row: get_energy_properties(
            row["switch"][:int(row["loop_end"]) + 1]),
        axis=1, result_type='expand',
        meta={0: 'str', 1: 'float64', 2: 'int64', 3: 'float64', 4: 'float64',
              5: 'str', 6: 'float64', 7: 'int64', 8: 'float64', 9: 'float64', 10: 'float64',
              11: 'float64', 12: 'float64'})

    triggers_ddf[[
        "stem_top_mfe_structure", "stem_top_mfe", "stem_top_mfe_matches_count",
        "stem_top_mfe_ensemble_defect", "stem_top_mfe_probability",
        "stem_top_centroid_structure", "stem_top_centroid_energy",
        "stem_top_centroid_matches_count",
        "stem_top_centroid_ensemble_defect", "stem_top_centroid_probability",
        "stem_top_partition_function", "stem_top_estimated_energy_mean",
        "stem_top_estimated_energy_std"
    ]] = triggers_ddf.apply(
        lambda row: get_energy_properties(
            row["switch"][int(row["stem_top_start"]):int(row["stem_top_end"]) + 1]),
        axis=1, result_type='expand',
        meta={0: 'str', 1: 'float64', 2: 'int64', 3: 'float64', 4: 'float64',
              5: 'str', 6: 'float64', 7: 'int64', 8: 'float64', 9: 'float64', 10: 'float64',
              11: 'float64', 12: 'float64'})


    triggers_ddf[[
        "trigger_mfe_structure", "trigger_mfe", "trigger_mfe_matches_count",
        "trigger_mfe_ensemble_defect", "trigger_mfe_probability",
        "trigger_centroid_structure", "trigger_centroid_energy", "trigger_centroid_matches_count",
        "trigger_centroid_ensemble_defect", "trigger_centroid_probability", "trigger_partition_function",
        "trigger_estimated_energy_mean", "trigger_estimated_energy_std"
    ]] = triggers_ddf.apply(
        lambda row: get_energy_properties(row["trigger"]),
        axis=1, result_type='expand',
        meta={0: 'str', 1: 'float64', 2: 'int64', 3: 'float64', 4: 'float64',
              5: 'str', 6: 'float64', 7: 'int64', 8: 'float64', 9: 'float64', 10: 'float64',
              11: 'float64', 12: 'float64'})

    triggers_df = triggers_ddf.compute()
    triggers_df.to_csv(OUTPUT_FILE_PATH)
