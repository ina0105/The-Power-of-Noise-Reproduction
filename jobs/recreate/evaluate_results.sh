#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=EvaluateResults
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:15:00
#SBATCH --output=output/test_evaluate_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

# Initialize conda for bash
source $(conda info --base)/etc/profile.d/conda.sh

cd $HOME/The-Power-of-Noise
conda activate power_of_noise

# Configuration
LLM_ID="meta-llama/Llama-2-7b-chat-hf"
OUTPUT_DIR="data/gen_res"
USE_TEST="False"
PROMPT_TYPE="classic"
USE_ADORE="False"
GET_DOCUMENTS_WITHOUT_ANSWER="True"

# List of NUM_DOCUMENTS to evaluate
# Modify this list to specify which document counts to evaluate
NUM_DOCUMENTS_LIST=(9 11)

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

# Function to evaluate experiment
evaluate_experiment() {
    local num_docs=$1
    local scenario=$2
    local gold_position=$3
    local use_random=$4
    local random_str=$5
    
    echo "=========================================="
    echo "Evaluating: ${scenario} scenario with ${random_str} documents"
    echo "Number of documents: ${num_docs}"
    echo "Gold position: ${gold_position}"
    echo "Use random: ${use_random}"
    echo "=========================================="
    
    python src/read_generation_results.py \
        --output_dir ${OUTPUT_DIR} \
        --llm_id ${LLM_ID} \
        --use_test ${USE_TEST} \
        --prompt_type ${PROMPT_TYPE} \
        --use_random ${use_random} \
        --use_adore ${USE_ADORE} \
        --gold_position ${gold_position} \
        --num_documents_in_context ${num_docs} \
        --get_documents_without_answer ${GET_DOCUMENTS_WITHOUT_ANSWER}
    
    echo ""
}



# Loop over each NUM_DOCUMENTS value
for NUM_DOCUMENTS in "${NUM_DOCUMENTS_LIST[@]}"; do
    echo "=========================================="
    echo "Evaluating Results for ${NUM_DOCUMENTS} documents"
    echo "=========================================="
    
    # Calculate gold positions for this number of documents
    FAR_POSITION=$(calculate_gold_position "far" $NUM_DOCUMENTS)
    MID_POSITION=$(calculate_gold_position "mid" $NUM_DOCUMENTS)
    NEAR_POSITION=$(calculate_gold_position "near" $NUM_DOCUMENTS)
    
    echo "Gold positions: far=${FAR_POSITION}, mid=${MID_POSITION}, near=${NEAR_POSITION}"
    echo ""
    
    # Table 1: Far, Mid, Near scenarios with Random documents
    echo "=========================================="
    echo "Evaluating Results for Table 2: Random Documents (${NUM_DOCUMENTS} docs)"
    echo "=========================================="
    
    evaluate_experiment $NUM_DOCUMENTS "far" $FAR_POSITION "True" "random"
    evaluate_experiment $NUM_DOCUMENTS "mid" $MID_POSITION "True" "random"
    evaluate_experiment $NUM_DOCUMENTS "near" $NEAR_POSITION "True" "random"
    
    # Table 2: Far, Mid, Near scenarios with Relevant/Distracting documents
    echo "=========================================="
    echo "Evaluating Results for Table 1: Relevant/Distracting Documents (${NUM_DOCUMENTS} docs)"
    echo "=========================================="
    
    evaluate_experiment $NUM_DOCUMENTS "far" $FAR_POSITION "False" "relevant"
    evaluate_experiment $NUM_DOCUMENTS "mid" $MID_POSITION "False" "relevant"
    evaluate_experiment $NUM_DOCUMENTS "near" $NEAR_POSITION "False" "relevant"
    
    echo ""
done

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
echo ""
echo "Summary: Accuracy values are printed above for each experiment."
echo "The detailed results are saved in: ${OUTPUT_DIR}/${LLM_ID#*/}/train/${PROMPT_TYPE}/contriever/"
echo "Look for files ending with 'all_extended.json' for detailed results."

