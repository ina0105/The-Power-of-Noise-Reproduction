#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Llama3UnquantizedExperiments
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:20:00
#SBATCH --output=output/llama3_unquantized_experiments_%A.out

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
# Make sure you have access to meta-llama/Llama-3-8B-Instruct model
if [ -f "$HOME/The-Power-of-Noise/.env" ]; then
    source "$HOME/The-Power-of-Noise/.env"
fi

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HuggingFace token not set!"
    echo "Please create a .env file in the project root with: HF_TOKEN=your_token_here"
    echo "You can get your token from: https://huggingface.co/settings/tokens"
    echo "Make sure you have access to meta-llama/Llama-3-8B-Instruct model"
    exit 1
fi

# Export HF_TOKEN as environment variable for Python scripts
export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

# Login to HuggingFace (alternative method if token is set)
huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || echo "Note: huggingface-cli login failed, but HF_TOKEN is set"

# Verify token is set (for debugging)
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN is set (length: ${#HF_TOKEN} characters)"
else
    echo "WARNING: HF_TOKEN is not set!"
fi

# Configuration
#meta-llama/Llama-3.2-1B-Instruct
LLM_ID="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_MAX_LENGTH=4096
OUTPUT_DIR="data/gen_res_prompt_ans_only_llama3_unquantized_og"
SAVE_EVERY=250
USE_VLLM="True"  # Set to "True" to use vLLM, "False" for HuggingFace
QUANTIZATION=""  # Empty for unquantized (no quantization)
TENSOR_PARALLEL_SIZE=1  # Number of GPUs for tensor parallelism (vLLM only)
DTYPE="bfloat16"  # Data type for vLLM: "bfloat16", "float16", "float32"

# Function to calculate batch size based on number of documents
calculate_batch_size() {
    local num_docs=$1
    
    if [ $num_docs -gt 9 ]; then
        echo 10
    elif [ $num_docs -gt 7 ]; then
        echo 20
    elif [ $num_docs -gt 5 ]; then
        echo 25
    else
        echo 40
    fi
}

# Function to run experiment
run_experiment() {
    local num_docs=$1
    local gold_position=$2
    
    local batch_size=$(calculate_batch_size $num_docs)
    
    echo "=========================================="
    echo "Running experiment: near scenario with random documents"
    echo "Number of documents: ${num_docs}"
    echo "Gold position: ${gold_position}"
    echo "Use random: True"
    echo "Batch size: ${batch_size}"
    echo "Model: ${LLM_ID} (Unquantized)"
    echo "Backend: ${USE_VLLM}"
    echo "=========================================="
    
    python src/generate_answers_llm.py \
        --output_dir ${OUTPUT_DIR} \
        --llm_id ${LLM_ID} \
        --model_max_length ${MODEL_MAX_LENGTH} \
        --load_full_corpus False \
        --use_random True \
        --use_adore False \
        --gold_position ${gold_position} \
        --num_documents_in_context ${num_docs} \
        --get_documents_without_answer True \
        --batch_size ${batch_size} \
        --save_every ${SAVE_EVERY} \
        --use_vllm ${USE_VLLM} \
        --quantization "${QUANTIZATION}" \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --dtype ${DTYPE}
    
    echo "Completed: near scenario with random documents (${num_docs} docs)"
    echo ""
}

# List of NUM_DOCUMENTS to run experiments for
NUM_DOCUMENTS_LIST=(11)

# Loop over each NUM_DOCUMENTS value - only near scenario with random documents
for NUM_DOCUMENTS in "${NUM_DOCUMENTS_LIST[@]}"; do
    # For near scenario, gold position is always the last position
    GOLD_POSITION=$((NUM_DOCUMENTS - 1))
    run_experiment $NUM_DOCUMENTS $GOLD_POSITION
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

