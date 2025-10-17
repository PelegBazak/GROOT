import json
import re
import subprocess

import numpy as np
import pandas as pd

INPUT_FILE_PATH = "/Data/full_df.csv"
OUTPUT_FILE_PATH = "/Data/full_df_with_rnaup.csv"
RNAUP_BINARY_NAME = "/bin/RNAup"
WORKING_DIRECTORY = ""
SEQUENCE_LENGTH = 23


def run_rna_up(sequence1, sequence2):
    try:
        return subprocess.check_output(
            [RNAUP_BINARY_NAME, "-b", "-d2", "--noLP", "-c", "'S'", "-o"],
            universal_newlines=True,
            input=f"{sequence1}&{sequence2}",
            text=True
        )
    except Exception as e:
        print(f"Error while running RNAup on {sequence1} and {sequence2}. Error: {e}")
        raise


def get_trigger_with_switch_mfe(trigger, switch) -> tuple[float, float, float, float, int]:
    rna_up_output = run_rna_up(trigger, switch)

    try:
        regex_result = re.search("\([0-9-.]+ = [0-9-.]+ \+ [0-9-+.]+ \+ ([0-9-+.]+\)|inf)",
                                 rna_up_output).group()
    except Exception as e:
        print(trigger)
        print(switch)
        raise e

    string_values = regex_result.replace("(", "").replace(")", "").replace("=", "").replace("+", "").split()
    values = [float(value) for value in string_values]
    total_free_energy = values[0]
    energy_from_duplex_formation = values[1]
    switch_opening_energy = values[2]
    trigger_opening_energy = values[3]
    secondary_structure = rna_up_output.split()[0]
    match_count = secondary_structure.count("(")
    return total_free_energy, energy_from_duplex_formation, switch_opening_energy, trigger_opening_energy, match_count


if __name__ == '__main__':
    triggers_df = pd.read_csv(INPUT_FILE_PATH)
    triggers_df[["trigger_with_switch_total_free_energy", "trigger_with_switch_duplex_formation_energy",
                 "switch_opening_energy", "trigger_opening_energy", "matches_count"]] = triggers_df.apply(
        lambda row: get_trigger_with_switch_mfe(row.trigger, row.switch), axis=1, result_type='expand'
    )

    triggers_df[["trigger_with_trigger_total_free_energy", "trigger_with_trigger_duplex_formation_energy",
                 "trigger_opening_energy_", "trigger_opening_energy_",
                 "trigger_with_trigger_matches_count"]] = triggers_df.apply(
        lambda row: get_trigger_with_switch_mfe(row.trigger, row.trigger), axis=1, result_type='expand'
    )

    triggers_df[["switch_with_switch_total_free_energy", "switch_with_switch_duplex_formation_energy",
                 "switch_opening_energy_", "trigger_opening_energy_",
                 "switch_with_switch_matches_count"]] = triggers_df.apply(
        lambda row: get_trigger_with_switch_mfe(row.trigger, row.trigger), axis=1, result_type='expand'
    )

    triggers_df = triggers_df.drop(columns=['switch_opening_energy_', 'trigger_opening_energy_'])
    triggers_df.to_csv(OUTPUT_FILE_PATH)
