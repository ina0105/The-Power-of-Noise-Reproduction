#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=RunPowerOfNoiseExperiments
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:30:00
#SBATCH --output=output/slurm_output_experiments_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

# Initialize conda for bash
source $(conda info --base)/etc/profile.d/conda.sh

cd $HOME/The-Power-of-Noise
conda activate power_of_noise

# Load HuggingFace token from .env file
# IMPORTANT: Create a .env file in the project root with: HF_TOKEN=your_token_here
# You can get your token from: https://huggingface.co/settings/tokens
# Make sure you have access to meta-llama/Llama-2-7b-chat-hf model
if [ -f "$HOME/The-Power-of-Noise/.env" ]; then
    source "$HOME/The-Power-of-Noise/.env"
fi

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HuggingFace token not set!"
    echo "Please create a .env file in the project root with: HF_TOKEN=your_token_here"
    echo "You can get your token from: https://huggingface.co/settings/tokens"
    echo "Make sure you have access to meta-llama/Llama-2-7b-chat-hf model"
    exit 1
fi

# Login to HuggingFace (alternative method if token is set)
huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || echo "Note: huggingface-cli login failed, but HF_TOKEN is set"

# Configuration
LLM_ID="meta-llama/Llama-2-7b-chat-hf"
MODEL_MAX_LENGTH=4096
OUTPUT_DIR="data/gen_res"
BATCH_SIZE=40
SAVE_EVERY=250

# Function to run experiment
run_experiment() {
    local scenario=$1
    local gold_position=$2
    local use_random=$3
    local random_str=$4
    
    echo "=========================================="
    echo "Running experiment: ${scenario} scenario with ${random_str} documents"
    echo "Gold position: ${gold_position}"
    echo "Use random: ${use_random}"
    echo "=========================================="
    
    python src/generate_answers_llm.py \
        --output_dir ${OUTPUT_DIR} \
        --llm_id ${LLM_ID} \
        --model_max_length ${MODEL_MAX_LENGTH} \
        --load_full_corpus False \
        --use_random ${use_random} \
        --use_adore False \
        --gold_position ${gold_position} \
        --num_documents_in_context ${NUM_DOCUMENTS} \
        --get_documents_without_answer True \
        --batch_size ${BATCH_SIZE} \
        --save_every ${SAVE_EVERY}
    
    echo "Completed: ${scenario} scenario with ${random_str} documents"
    echo ""
}

echo "=========================================="
echo "Starting experiments for Table 1: Relevant/Distracting Documents"
echo "=========================================="

# NUM_DOCUMENTS=9
# BATCH_SIZE=20
# # Far scenario with random documents (gold at position 0)
# run_experiment "near" 8 "False" "relevant"

# # Far scenario with relevant/distracting documents (gold at position 0)
# run_experiment "far" 0 "False" "relevant"

# # Mid scenario with relevant/distracting documents (gold at position 3)
# run_experiment "mid" 4 "False" "relevant"

# Near scenario with relevant/distracting documents (gold at position 6)
# run_experiment "near" 2 "False" "relevant"


# Run experiments for first two tables
# Table 1: Far, Mid, Near scenarios with Random documents
echo "=========================================="
echo "Starting experiments for Table 2: Random Documents"
echo "=========================================="

NUM_DOCUMENTS=13
BATCH_SIZE=10
# Far scenario with random documents (gold at position 0)
run_experiment "near" 12 "True" "random"
# Mid scenario with random documents (gold at position 3)
#run_experiment "mid" 1 "True" "random"

# Near scenario with random documents (gold at position 6)
# run_experiment "near" 2 "True" "random"



