import pandas as pd
from Bio.SeqUtils import gc_fraction

POTENTIAL_TRIGGERS_FILE_PATH = "/Data/full_df.csv"
OUTPUT_FILE_PATH = "/Data/full_df_with_gc_content.csv"


if __name__ == '__main__':    
    triggers_df = pd.read_csv(POTENTIAL_TRIGGERS_FILE_PATH, index_col=0)    
    triggers_df["switch_gc_content"] = triggers_df.apply(
        lambda row: gc_fraction(row["switch"]), axis=1)

    triggers_df["binding_site_gc_content"] = triggers_df.apply(
        lambda row: gc_fraction(
            row["switch"][int(row["trigger_binding_site_start"]):int(row["trigger_binding_site_end"]) + 1]
        ), axis=1)

    triggers_df["stem_gc_content"] = triggers_df.apply(
        lambda row: gc_fraction(
            row["switch"][int(row["stem_start"]):int(row["stem_end"]) + 1]
        ), axis=1)

    triggers_df["loop_gc_content"] = triggers_df.apply(
        lambda row: gc_fraction(
            row["switch"][int(row["loop_start"]):int(row["loop_end"]) + 1]
        ), axis=1)

    triggers_df["loop_to_stem_end_gc_content"] = triggers_df.apply(
        lambda row: gc_fraction(
            row["switch"][int(row["loop_start"]):int(row["stem_end"]) + 1]
        ), axis=1)

    triggers_df["loop_to_end_gc_content"] = triggers_df.apply(
        lambda row: gc_fraction(
            row["switch"][int(row["loop_start"]):],
        ), axis=1)

    triggers_df["stem_start_to_loop_end_gc_content"] = triggers_df.apply(
        lambda row: gc_fraction(
            row["switch"][int(row["stem_start"]):int(row["loop_end"]) + 1],
        ), axis=1)

    triggers_df["start_to_loop_end_gc_content"] = triggers_df.apply(
        lambda row: gc_fraction(
            row["switch"][:int(row["loop_end"]) + 1],
        ), axis=1)

    triggers_df["stem_top_gc_content"] = triggers_df.apply(
        lambda row: gc_fraction(
            row["switch"][int(row["stem_top_start"]):int(row["stem_top_end"]) + 1],
        ), axis=1)

    triggers_df["trigger_gc_content"] = triggers_df.apply(
        lambda row: gc_fraction(
            row["trigger"],
        ), axis=1)

    triggers_df.to_csv(OUTPUT_FILE_PATH)
