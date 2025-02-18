#!/bin/bash -l

#SBATCH --job-name=RAW_MEDQA_TEST
#SBATCH --output=slurm_out/%A_raw_medqa_test.out
#SBATCH --error=slurm_out/%A_raw_medqa_test.err
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:2
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=coen.arrow@research.uwa.edu.au

printf "Starting job $SLURM_JOB_NAME, ID $SLURM_JOB_ID\n\n" 

 # Load the relevant modules
module load cuda/12.4
module load Anaconda3/2024.06

# leave in, it lists the environment loaded by the modules
module list

# Set project directory
PROJECT_DIR=/group/pgh004/carrow/repo/MedLLM/PMC-LLaMA
cd $PROJECT_DIR
# load the conda environment
conda activate ./env
SCRIPT="main.py"
CONFIG_PATH="configs/medqa-test_raw.yml"

printf " \n Run on dataset \n Started at $(date) \n \n " 
start_time=$(date +%s)

# Run the Python script
python "${PROJECT_DIR}/${SCRIPT}" --config_filepath "${PROJECT_DIR}/${CONFIG_PATH}"
end_time=$(date +%s)
runtime=$((end_time-start_time))
printf " \n Finished at $(date) \n " 
printf " \n Total runtime: $runtime seconds \n " 