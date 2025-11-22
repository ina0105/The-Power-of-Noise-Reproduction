#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=RunPowerOfNoiseExperiments
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=06:00:00
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
BATCH_SIZE=15
SAVE_EVERY=250

# List of NUM_DOCUMENTS to run experiments for
# Modify this list to specify which document counts to test
NUM_DOCUMENTS_LIST=(11)

# Function to calculate gold position based on scenario and num_docs
calculate_gold_position() {
    local scenario=$1
    local num_docs=$2
    
    case $scenario in
        "far")
            echo 0
            ;;
        "mid")
            echo $((num_docs / 2))
            ;;
        "near")
            echo $((num_docs - 1))
            ;;
        *)
            echo "Unknown scenario: $scenario" >&2
            exit 1
            ;;
    esac
}

# Function to run experiment
run_experiment() {
    local num_docs=$1
    local scenario=$2
    local gold_position=$3
    local use_random=$4
    local random_str=$5
    
    echo "=========================================="
    echo "Running experiment: ${scenario} scenario with ${random_str} documents"
    echo "Number of documents: ${num_docs}"
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
        --num_documents_in_context ${num_docs} \
        --get_documents_without_answer True \
        --batch_size ${BATCH_SIZE} \
        --save_every ${SAVE_EVERY}
    
    echo "Completed: ${scenario} scenario with ${random_str} documents (${num_docs} docs)"
    echo ""
}



# Loop over each NUM_DOCUMENTS value
for NUM_DOCUMENTS in "${NUM_DOCUMENTS_LIST[@]}"; do
    echo "=========================================="
    echo "Starting experiments for ${NUM_DOCUMENTS} documents"
    echo "=========================================="
    
    # Calculate gold positions for this number of documents
    FAR_POSITION=$(calculate_gold_position "far" $NUM_DOCUMENTS)
    MID_POSITION=$(calculate_gold_position "mid" $NUM_DOCUMENTS)
    NEAR_POSITION=$(calculate_gold_position "near" $NUM_DOCUMENTS)
    
    echo "Gold positions: far=${FAR_POSITION}, mid=${MID_POSITION}, near=${NEAR_POSITION}"
    echo ""
    
    # Table 1: Far, Mid, Near scenarios with Random documents
    echo "=========================================="
    echo "Table 1: Random Documents (${NUM_DOCUMENTS} docs)"
    echo "=========================================="
    
    run_experiment $NUM_DOCUMENTS "far" $FAR_POSITION "True" "random"
    run_experiment $NUM_DOCUMENTS "mid" $MID_POSITION "True" "random"
    run_experiment $NUM_DOCUMENTS "near" $NEAR_POSITION "True" "random"
    
    # Table 2: Far, Mid, Near scenarios with Relevant/Distracting documents
    echo "=========================================="
    echo "Table 2: Relevant/Distracting Documents (${NUM_DOCUMENTS} docs)"
    echo "=========================================="
    
    run_experiment $NUM_DOCUMENTS "far" $FAR_POSITION "False" "relevant"
    run_experiment $NUM_DOCUMENTS "mid" $MID_POSITION "False" "relevant"
    run_experiment $NUM_DOCUMENTS "near" $NEAR_POSITION "False" "relevant"
    
    echo ""
done



echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

