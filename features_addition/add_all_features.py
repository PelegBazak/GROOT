import sys

sys.path.insert(1, '..')
from utils.pbs_job_submission import submit_job
from utils.slurm_job_submission import submit_cpu_job

if __name__ == '__main__':
    submit_job(
        "rna_fold_features",
        f"python /GROOT/features_addition/get_rna_fold_features.py",
        ram=2,
        queue_name=""
    )

    submit_job(
        "rna_up",
            f"python /GROOT/features_addition/adding_trigger_to_switch_energy_properties.py",
        ram=20,
        cores_count=5,
        queue_name=""
    )

    submit_job(
        "rna_cofold",
        f"python /GROOT/features_addition/add_trigger_to_full_switch_energy_properties.py",
        ram=20,
        cores_count=5,
        queue_name=""
    )

    submit_job(
        "gc_content",
        f"python /GROOT/features_addition/add_gc_content.py",
        ram=2,
        queue_name=""
    )

    submit_job(
        "nupack",
        f"python /GROOT/features_addition/adding_nupack_properties_on_healthy_gene.py",
        ram=10,
        cores_count=5,
        queue_name=""
    )

    submit_job(
        "translation_initiation",
        f"python /GROOT/features_addition/add_translation_initiation_energy_features.py",
        ram=20,
        cores_count=5,
        queue_name=""
    )

    submit_job(
        "ran_eval",
        f"python /GROOT/features_addition/adding_rna_eval_results.py",
        ram=45,
        cores_count=5,
        queue_name=""
    )

    submit_job(
        "risearch",
        f"python /GROOT/features_addition/adding_risearch_results.py",
        ram=45,
        cores_count=5,
        queue_name=""
    )

    submit_cpu_job(
        f"eterna_fold",
        f"python /GROOT/features_addition/adding_eternafold_features.py",
        ram=60,
        partition="",
        account=""
    )

    submit_cpu_job(
        f"pyfeat",
        f"python /GROOT/features_addition/add_nucleotide_patterns_features.py",
        ram=240,
        partition="",
        account=""
    )

    submit_job(
        f"rrna_profiles",
        f"python /GROOT/features_addition/adding_rRNA_profiles.py",
        ram=45,
        queue_name=""
    )
