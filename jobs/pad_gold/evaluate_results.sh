#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=EvaluateResultsPadGold
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:15:00
#SBATCH --output=output/test_evaluate_pad_gold_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

# Initialize conda for bash
source $(conda info --base)/etc/profile.d/conda.sh

cd $HOME/The-Power-of-Noise
conda activate power_of_noise

# Configuration
LLM_ID="meta-llama/Llama-2-7b-chat-hf"
OUTPUT_DIR="data/gen_res_pad_gold"
USE_TEST="False"
PROMPT_TYPE="classic"
USE_ADORE="False"
GET_DOCUMENTS_WITHOUT_ANSWER="True"

# List of NUM_DOCUMENTS to evaluate
# Modify this list to specify which document counts to evaluate
NUM_DOCUMENTS_LIST=(1 2 3 5 7 9 11 13 15)

# Function to evaluate experiment
evaluate_experiment() {
    local num_docs=$1
    local gold_position=$2
    
    echo "=========================================="
    echo "Evaluating: near scenario with random documents (pad_gold)"
    echo "Number of documents: ${num_docs}"
    echo "Gold position: ${gold_position}"
    echo "Use random: True"
    echo "=========================================="
    
    python src/read_generation_results.py \
        --output_dir ${OUTPUT_DIR} \
        --llm_id ${LLM_ID} \
        --use_test ${USE_TEST} \
        --prompt_type ${PROMPT_TYPE} \
        --use_random True \
        --use_adore ${USE_ADORE} \
        --gold_position ${gold_position} \
        --num_documents_in_context ${num_docs} \
        --get_documents_without_answer ${GET_DOCUMENTS_WITHOUT_ANSWER} \
        --pad_gold True \
        --overwrite True
    
    echo ""
}

# Loop over each NUM_DOCUMENTS value - only near scenario with random documents
for NUM_DOCUMENTS in "${NUM_DOCUMENTS_LIST[@]}"; do
    echo "=========================================="
    echo "Evaluating Results for ${NUM_DOCUMENTS} documents (near scenario, random, pad_gold)"
    echo "=========================================="
    
    # For near scenario, gold position is always the last position
    GOLD_POSITION=$((NUM_DOCUMENTS - 1))
    
    echo "Gold position: ${GOLD_POSITION}"
    echo ""
    
    evaluate_experiment $NUM_DOCUMENTS $GOLD_POSITION
    
    echo ""
done

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
echo ""
echo "Summary: Accuracy values are printed above for each experiment."
echo "The detailed results are saved in: ${OUTPUT_DIR}/${LLM_ID#*/}/train/${PROMPT_TYPE}/contriever/"
echo "Look for files ending with 'all_extended.json' for detailed results."

