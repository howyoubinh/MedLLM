#!/bin/bash -l

#SBATCH --job-name=13B_with_entities
#SBATCH --output=hpc_scripts/out/QA_Inference_13B.out
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:2
printf "Starting job $SLURM_JOB_NAME, ID $SLURM_JOB_ID\n\n" 

 # Load the relevant modules
module load cuda/11.6

# leave in, it lists the environment loaded by the modules
module list

# Set project directory
PROJECT_DIR=/group/pgh004/carrow/repo/MedLLM/PMC-LLaMA
cd $PROJECT_DIR
# load the conda environment
conda activate ./env
# echo the loaded environment
conda env list
SCRIPT=medqa_inference.py

printf " \n Started at $(date) \n \n " 
start_time=$(date +%s)

# Run the three python scripts below in parallel
python ${SCRIPT} --model-name-or-path axiong/PMC_LLaMA_13B --write-dir inferenced_result_dir/medqa_with_entities --models-dir "models" --data-dir "data" --extract-entities

end_time=$(date +%s)
runtime=$((end_time-start_time))
printf " \n Finished at $(date) \n " 
printf " \n Total runtime: $runtime seconds \n " 