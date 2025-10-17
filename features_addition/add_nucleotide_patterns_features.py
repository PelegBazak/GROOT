import os

import numpy as np
import pandas as pd

import sys
sys.path.insert(1, '/PyFeat/Codes/')
import generateFeatures


DATA_DIR = "/Data/"


args = {
    'sequenceType': 'RNA',
    'kGap': 2, # default is 5
    'kTuple': 3,
    'pseudoKNC': 1,
    'zCurve': 1,
    'cumulativeSkew': 1,
    'gcContent': 0,
    'atgcRatio': 1,
    'monoMono': 1,
    'monoDi': 1,
    'monoTri': 1,
    'diMono': 1,
    'diDi': 1,
    'diTri': 1,
    'triMono': 1,
    'triDi': 1
}


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == '__main__':
    triggers_df = pd.read_csv(
        os.path.join(DATA_DIR, "full_df.csv"),
        index_col=0).reset_index(drop=True)

    T = generateFeatures.gF(dotdict(args), triggers_df.switch, triggers_df.switch)  # some dummy column as y
    triggers_df_with_nucleotides_features = pd.DataFrame(np.column_stack([triggers_df, T[:, :-1]]))  # last column is the "label"

    additional_features = [f'zCurve_{i}' for i in range(1, 4)] + [f'cumulativeSkew_{i}' for i in range(1, 3)] \
                          + ['atgcRatio'] + [f'pseudoKNC_{i}' for i in range(1, 1 + 4 + 16 + 64)] \
                          + [f'monoMonoKGap_{i}' for i in range(1, 33)] + [f'monoDiKGap_{i}' for i in range(1, 129)] \
                          + [f'monoTriKGap_{i}' for i in range(1, 513)] \
                          + [f'diMonoKGap_{i}' for i in range(1, 129)] + [f'diDiKGap_{i}' for i in range(1, 513)] \
                          + [f'diTriKGap_{i}' for i in range(1, 2049)] \
                          + [f'triMonoKGap_{i}' for i in range(1, 513)] + [f'triDiKGap_{i}' for i in range(1, 2049)]

    triggers_df_with_nucleotides_features.columns = list(triggers_df.columns) + additional_features

    triggers_df_with_nucleotides_features.to_csv(
        os.path.join(DATA_DIR, "full_df_with_nucleotide_patterns_features.csv"))
