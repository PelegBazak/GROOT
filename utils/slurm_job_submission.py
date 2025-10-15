import subprocess


def submit_cpu_job(job_name: str, command: str, partition: str, account: str, ram=120, cores_count=5):
    job_script = f"""#!/bin/bash
    
#SBATCH --job-name={job_name}             # Job name
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task={cores_count}             # Number of CPU cores per task
#SBATCH --mem={ram}G
#SBATCH --output={job_name}.log        # Standard output and error log (%j expands to jobId)
#SBATCH --time=7-00:00:00        # 7 days

cd /GROOT
source ".venv/bin/activate"
{command}
    """

    subprocess.run(["sbatch", "-p", partition, "-A", account], input=job_script, text=True)


def submit_gpu_job(job_name: str, command: str, partition: str, ram=60, cores_count=5):
    job_script = f"""#!/bin/bash

#SBATCH --job-name={job_name}             # Job name
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task={cores_count}             # Number of CPU cores per task
#SBATCH --mem={ram}G
#SBATCH --gres=gpu:1
#SBATCH --output={job_name}.log        # Standard output and error log (%j expands to jobId)
#SBATCH --time=7-00:00:00        # 7 days

cd /GROOT
source ".venv/bin/activate"
{command}
    """

    subprocess.run(["sbatch", "-p", partition], input=job_script, text=True)
