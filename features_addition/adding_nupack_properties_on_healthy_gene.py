import json
import logging

import numpy as np
import pandas as pd
from nupack import *

config.threads = 4

import sys
sys.path.insert(1, '..')
from logging_utils import get_logger

INPUT_FILE_PATH = "/Data/full_df.csv"
OUTPUT_FILE_PATH = "/Data/full_df_with_nupack.csv"
WORKING_DIRECTORY = ""
LOGGER_NAME = "feature_adding_logger"
LOG_FILE = "run.log"

logger = get_logger(LOGGER_NAME, os.path.join(WORKING_DIRECTORY, LOG_FILE))


def get_trigger_switch_complex_concentration(design_result) -> float:
    target_concentration = sum([c for name, c in design_result["t1"].complex_concentrations.items()])

    try:
        predicted_concentration = [(name, c) for name, c in design_result["t1"].complex_concentrations.items()
                                   if str(name) == "<Complex (switch+trigger)>"][0][1]
    except IndexError as e:
        predicted_concentration = [(name, c) for name, c in design_result["t1"].complex_concentrations.items()
                                   if str(name) == "<Complex (trigger+switch)>"][0][1]

    return predicted_concentration / target_concentration


def get_nupack_properties(switch, trigger_strand):
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    try:
        logger.info("stated nupack properties")

        # specify strands
        trigger_strand = Strand(trigger_strand, name='trigger', material="rna")
        switch_strand = Strand(switch, name='switch', material="rna")

        # specify tubes
        tube = Tube(strands={trigger_strand: 1e-8, switch_strand: 1e-8}, complexes=SetSpec(max_size=2), name='t1')
        tube_results = tube_analysis(tubes=[tube], model=Model(material='rna', celsius=37))
        complex_concentration = get_trigger_switch_complex_concentration(tube_results)
        logger.info("finished nupack properties")
    except Exception as e:
        logger.error(e)
        raise
    return complex_concentration


if __name__ == '__main__':
    triggers_df = pd.read_csv(INPUT_FILE_PATH)
    logger.info("finished loading data")
    triggers_df["nupack_predicted_concentration"] = triggers_df.apply(
        lambda row: get_nupack_properties(row.trigger, row.switch), axis=1)
    logger.info("saving results")
    triggers_df.to_csv(OUTPUT_FILE_PATH)
