import os
import re
import subprocess
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = "/Data/"
INPUT_FILE_NAME = "full_df.csv"
OUTPUT_FILE_NAME = "full_df_with_bulge_features.csv"
RNAPLEX_BINARY_NAME = "/bin/RNAplex"
LABEL_ENCODINGS = {
    "A": 0,
    "C": 1,
    "G": 2,
    "U": 3
}


def run_rna_plex(sequence1, sequence2):
    try:
        return subprocess.check_output(
            [RNAPLEX_BINARY_NAME],
            universal_newlines=True,
            input=f"{sequence1}\n{sequence2}",
            text=True
        )
    except Exception as e:
        print(f"Error while running RNAplex on {sequence1} and {sequence2}. Error: {e}")
        raise


def get_hybridiztion_energy(sequence1, sequence2) -> float:
    rna_plex_output = run_rna_plex(sequence1, sequence2)

    if rna_plex_output == "\n":
        return 0

    try:
        regex_result = re.search("\(\s*[0-9-.]+\s*\)", rna_plex_output).group()
    except Exception as e:
        print(f"Error while parsing RNAplex result on {sequence1} and {sequence2}. Error: {e}")
        raise

    return float(regex_result.replace("(", "").replace(")", "").strip())

def get_hybridiztion_matches_count(sequence1, sequence2) -> float:
    rna_plex_output = run_rna_plex(sequence1, sequence2)

    if rna_plex_output == "\n":
        return 0

    return rna_plex_output.split()[0].count('(')

def get_dna_onehot_encoding(seq):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # get sequence into an array
        seq_array = np.array(list(seq))

        # integer encode the sequence
        integer_encoded_seq = np.array([LABEL_ENCODINGS[item] for item in seq_array])

        # one hot the sequence
        onehot_encoder = OneHotEncoder(sparse_output=False, categories=[np.array([0, 1, 2, 3])])
        # reshape because that's what OneHotEncoder likes
        integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
        onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)

        return onehot_encoded_seq.reshape(1, -1).squeeze()


if __name__ == '__main__':
    triggers_df = pd.read_csv(
        os.path.join(DATA_DIR, INPUT_FILE_NAME), index_col=0, usecols=lambda x: x not in [
            'risearch_results', 'rrna_mrna_interaction_strength_profile', 'eterna_fold_bppm'
        ]
    )

    triggers_df["stem_sides_hybridization_energy"] = triggers_df.apply(
        lambda row: get_hybridiztion_energy(
            row.switch[int(row["stem_start"]):int(row["loop_start"])],
            row.switch[int(row["loop_end"] + 1):int(row["stem_end"] + 1)]), axis=1
    )

    triggers_df["stem_to_trigger_hybridization_energy"] = triggers_df.apply(
        lambda row: get_hybridiztion_energy(
            row.switch[int(row["stem_start"]):int(row["loop_start"])],
            row.trigger[:len(row.switch[int(row["stem_start"]):int(row["loop_start"])])]), axis=1
    )

    triggers_df["stem_vs_trigger_hybridization_energy_diff"] = triggers_df["stem_sides_hybridization_energy"] - triggers_df["stem_to_trigger_hybridization_energy"]

    triggers_df["bulge_sides_hybridization_energy"] = triggers_df.apply(
        lambda row: get_hybridiztion_energy(
            row.switch[int(row["stem_top_start"]) - 3:int(row["stem_top_start"])],
            row.switch[int(row["stem_top_end"]) + 1:int(row["stem_top_end"]) + 4]), axis=1
    )

    triggers_df["bulge_to_trigger_hybridization_energy"] = triggers_df.apply(
        lambda row: get_hybridiztion_energy(
            row.switch[int(row["stem_top_start"]) - 3:int(row["stem_top_start"])],
            row.trigger[int(row["stem_top_start"]) - int(row["stem_start"]) - 3:int(row["stem_top_start"]) - int(row["stem_start"])]), axis=1
    )

    triggers_df["bulge_vs_trigger_hybridization_energy_diff"] = triggers_df["bulge_sides_hybridization_energy"] - triggers_df["bulge_to_trigger_hybridization_energy"]

    triggers_df["bulge_sides_hybridization_matches_count"] = triggers_df.apply(
        lambda row: get_hybridiztion_matches_count(
            row.switch[int(row["stem_top_start"]) - 3:int(row["stem_top_start"])],
            row.switch[int(row["stem_top_end"]) + 1:int(row["stem_top_end"]) + 4]), axis=1
    )

    triggers_df["bulge_to_trigger_hybridization_matches_count"] = triggers_df.apply(
        lambda row: get_hybridiztion_matches_count(
            row.switch[int(row["stem_top_start"]) - 3:int(row["stem_top_start"])],
            row.trigger[int(row["stem_top_start"]) - int(row["stem_start"]) - 3:int(row["stem_top_start"]) - int(
                row["stem_start"])]), axis=1
    )

    triggers_df[[f"bulge_{i}_{key}" for i in range(3) for key in LABEL_ENCODINGS.keys()]] = triggers_df.apply(
        lambda row: get_dna_onehot_encoding(row.switch[int(row["stem_top_start"]) - 3:int(row["stem_top_start"])]),
        axis=1, result_type='expand'
    )
    triggers_df[[f"trigger_corresponding_to_bulge_{i}_{key}" for i in range(3) for key in
                 LABEL_ENCODINGS.keys()]] = triggers_df.apply(
        lambda row: get_dna_onehot_encoding(
            row.trigger[int(row["stem_top_start"]) - int(row["stem_start"]) - 3:int(row["stem_top_start"]) - int(
                row["stem_start"])]),
        axis=1, result_type='expand'
    )

    triggers_df.to_csv(os.path.join(DATA_DIR, OUTPUT_FILE_NAME))
