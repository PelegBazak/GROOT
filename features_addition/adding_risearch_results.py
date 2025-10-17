import logging
import os
import random
import re
import subprocess
import sys
from typing import Dict, List

import dask.dataframe as dd
import pandas as pd
from Bio.Seq import Seq
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

sys.path.insert(1, '..')
from logging_utils import get_logger


WORKING_DIRECTORY = ""
DATA_DIR = "/Data/"
RISEARCH_BINARY_NAME = "/bin/RIsearch1"
LOGGER_NAME = "risreach_logger"

logger = get_logger(LOGGER_NAME, os.path.join(WORKING_DIRECTORY, "riseach_logger.log"))


def get_mfe_scores(result: str, query_name: str) -> List[List[float]]:
    mfe_results = []

    for gene_result in result.split(f"\n\nquery {query_name}")[1:]:
        stripped_result = gene_result.strip()
        regex_results = re.findall("Free energy \[kcal/mol\]: [0-9-.]+ ", stripped_result)
        mfe_results.append(
            [float(regex_result.replace('Free energy [kcal/mol]: ', '').strip()) for regex_result in regex_results])

    return mfe_results

def get_trigger_mfe_scores_by_risearch(switch: str, trigger: str, name_to_sequence: Dict[str, str]):
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.info(f"started evaluating a single trigger: {trigger}")
    hash = random.getrandbits(64)

    with open(f"target-{hash}.fa", "w") as f:
        for name, sequence in name_to_sequence.items():
            f.write(">" + str(name) + "\n" + sequence + "\n")

    # binding site
    with open(f"query-{hash}.fa", "w") as f:
        f.write(">binding_site" + "\n" + str(Seq(trigger).reverse_complement_rna()) + "\n")

    result_binding_site = subprocess.check_output(
        [RISEARCH_BINARY_NAME, "-q", f"query-{hash}.fa", "-t", f"target-{hash}.fa", "-s", "2400", "-n", "20", "-d", "30"],
        universal_newlines=True,
        text=True
    )
    mfe_scores_binding_site = get_mfe_scores(result_binding_site, "binding_site")

    # switch
    with open(f"query-{hash}.fa", "w") as f:
        f.write(">switch" + "\n" + switch + "\n")

    result_switch = subprocess.check_output(
        [RISEARCH_BINARY_NAME, "-q", f"query-{hash}.fa", "-t", f"target-{hash}.fa", "-s", "2400", "-n", "20", "-d", "30"],
        universal_newlines=True,
        text=True
    )
    mfe_scores_switch = get_mfe_scores(result_switch, "switch")

    # trigger
    with open(f"query-{hash}.fa", "w") as f:
        f.write(">trigger" + "\n" + trigger + "\n")

    result_trigger = subprocess.check_output(
        [RISEARCH_BINARY_NAME, "-q", f"query-{hash}.fa", "-t", f"target-{hash}.fa", "-s", "2400", "-n", "20", "-d", "30"],
        universal_newlines=True,
        text=True
    )
    mfe_scores_trigger = get_mfe_scores(result_trigger, "trigger")

    os.remove(f"target-{hash}.fa")
    os.remove(f"query-{hash}.fa")
    logger.info(f"finished evaluating a single trigger: {trigger}")
    return mfe_scores_trigger, mfe_scores_switch, mfe_scores_binding_site


if __name__ == '__main__':
    all_triggers_df = pd.read_csv(os.path.join(DATA_DIR, "full_df.csv"), index_col=0,
        usecols=lambda x: x not in ['risearch_results', 'rrna_mrna_interaction_strength_profile', 'eterna_fold_bppm']
    )[["trigger", "switch"]]

    fpkm_df = pd.read_csv(os.path.join(DATA_DIR, "GSE278616_fpkm_all_samples.tsv"), sep="\t", index_col=0)
    fpkm_df = fpkm_df.mean(axis=1).reset_index().rename(columns={
        0: "mean_fpkm",
        "Gene_ID": "gene_id"
    })
    fpkm_df.gene_id = fpkm_df.gene_id.apply(lambda gene_id: gene_id[5:])

    genes_df = pd.read_csv(os.path.join(DATA_DIR, "Copy-of-All-genes-of-E.-coli-BL21(DE3) - ECD accession.tsv"),
                           sep="\t").drop(columns=["Fragments"])
    genes_df = genes_df.drop(columns=["Accession-1"])

    genes_with_expression_levels = genes_df.merge(fpkm_df, left_on="Gene Name", right_on="gene_id")

    highly_expressed_genes = genes_with_expression_levels[
        genes_with_expression_levels.mean_fpkm >= genes_with_expression_levels.mean_fpkm.quantile(0.8)]
    highly_expressed_genes["Sequence - DNA sequence"] = highly_expressed_genes["Sequence - DNA sequence"].apply(
        lambda sequence: sequence.upper().replace("T", "U"))

    genes_with_expression_levels["Sequence - DNA sequence"] = genes_with_expression_levels["Sequence - DNA sequence"].apply(
        lambda sequence: sequence.upper().replace("T", "U"))

    logger.info(f"creating cluster")

    cluster = SLURMCluster(
        cores=2,
        processes=1,
        memory="60GB",
        account="",
        walltime="3-00:00:00",
        queue="",

    )

    cluster.scale(jobs=10)
    client = Client(cluster)
    client.forward_logging()
    switches_ddf = dd.from_pandas(all_triggers_df, npartitions=10*3)

    switches_ddf[[
        "risearch_results_trigger",
        "risearch_results_switch", "risearch_results_binding_site"]] = switches_ddf.apply(
        lambda row: get_trigger_mfe_scores_by_risearch(row.switch, row.trigger, dict(
            enumerate(genes_with_expression_levels['Sequence - DNA sequence'].to_list()))),
        axis=1, result_type='expand',
        meta={0: 'str', 1: 'str', 2: 'str'}
    )
    switches_ddf[[
        "risearch_results_trigger_highly_expressed_genes",
        "risearch_results_switch_highly_expressed_genes",
        "risearch_results_binding_site_highly_expressed_genes"]] = switches_ddf.apply(
        lambda row: get_trigger_mfe_scores_by_risearch(row.switch, row.trigger, dict(
            enumerate(highly_expressed_genes['Sequence - DNA sequence'].to_list()))),
        axis=1, result_type='expand',
        meta={0: 'str', 1: 'str', 2: 'str'}
    )

    logger.info(f"start evaluating triggers")
    switches_df = switches_ddf.compute()

    switches_df.to_csv(os.path.join(DATA_DIR, "full_df_with_risearch_analysis.csv"))

