import subprocess


def submit_job(job_name: str, command: str, queue_name, ram=10, hours_to_run=24, cores_count=1):
    job_script = f"""#!/bin/bash
    #PBS -q {queue_name}
    #PBS -l nodes=1:ppn=1
    #PBS -l walltime={hours_to_run}:00:00
    #PBS -l pmem={ram}gb,mem={ram}gb,pvmem={ram}gb,vmem={ram}gb,ncpus={cores_count}
    #PBS -N {job_name}
    #PBS -V
    
    source "/venv/bin/activate"
    {command}
    """

    subprocess.run(["qsub"], input=job_script, text=True)