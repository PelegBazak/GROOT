import json
import re
import subprocess

import numpy as np
import pandas as pd

INPUT_FILE_PATH = "/Data/full_df.csv"
OUTPUT_FILE_PATH = \
    "/Data/full_df_with_cofold.csv"
RNACOFOLD_BINARY_NAME = "/bin/RNAcofold"
WORKING_DIRECTORY = ""
SEQUENCE_LENGTH = 23


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


def get_trigger_with_switch_mfe(trigger, switch) -> tuple[float, int]:
    rna_cofold_output = run_rna_cofold(trigger, switch)
    regex_result = re.search("\([0-9-]+[0-9.]*\)", rna_cofold_output).group()
    mfe = float(regex_result.replace('(', '').replace(')', ''))
    secondary_structure = rna_cofold_output.split()[1]
    match_count = secondary_structure.count("(")
    return mfe, match_count


if __name__ == '__main__':
    triggers_df = pd.read_csv(INPUT_FILE_PATH)
    triggers_df[["trigger_with_switch_mfe_cofold",
                 "trigger_with_switch_matches_count_cofold"]] = triggers_df.apply(
        lambda row: get_trigger_with_switch_mfe(row["trigger"], row["switch"]), axis=1, result_type='expand')

    triggers_df.to_csv(OUTPUT_FILE_PATH)
