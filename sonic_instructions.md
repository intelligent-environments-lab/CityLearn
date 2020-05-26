# How to access and use the Sonic HPC for train jobs

## Access

1. Use ssh to navigate to: `ssh -J user_id@resit-ssh.ucd.ie user_id@sonic.ucd.ie`
2. Use the password provided by the UCD IT Services

## Prepare

1. Use command: `touch setup_cl.sh`
2. Use command: `vim setup_cl.sh` and copy paste the following code:

 ```bash
 #!/bin/bash -l
 mkdir rl
 cd rl
 git clone https://github.com/anjukan/CityLearn
 cd CityLearn
 module load anaconda
 conda env create -f environment.yml
 conda init bash
 ```

3. Use `ESC :wq ENTER` to save the file in vim
4. Use command: `chmod a+x setup_cl.sh`
5. Use command: `./setup_cl.sh`
6. Use command: `cp rl/CityLearn/run_train.sh .`
7. Use command: `chmod a+x run_train.sh`
8. (Optional) Modify the job parameters in `run_train.sh` using vim

## Run

1. Use command `sbatch --partition=gpu run_train.sh` to schedule the job
2. Use command `squeue --user=user_id` to see your scheduled jobs
3. Use command `cat slurm_job_id.out` to see the standard output from that job
4. (Optional) Use `scancel job_id` to cancel the job
