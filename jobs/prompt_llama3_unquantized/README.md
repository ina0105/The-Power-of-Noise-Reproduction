# Llama 3 Unquantized Experiments

This folder contains job scripts for running experiments with **Llama 3 8B Instruct** model in **unquantized** (full precision) mode using vLLM.

## Configuration

- **Model**: `meta-llama/Llama-3-8B-Instruct`
- **Backend**: vLLM (`USE_VLLM="True"`)
- **Quantization**: None (unquantized, `QUANTIZATION=""`)
- **Data Type**: `bfloat16`
- **Tensor Parallelism**: 1 GPU

## Files

1. **run_experiments.sh**: Runs generation experiments for different numbers of documents (1, 2, 9, 11, 13)
2. **evaluate_results.sh**: Evaluates the generated results and computes accuracy

## Usage

### Running Experiments

```bash
cd /home/scur2158/The-Power-of-Noise/jobs/prompt_llama3_unquantized
sbatch run_experiments.sh
```

### Evaluating Results

```bash
cd /home/scur2158/The-Power-of-Noise/jobs/prompt_llama3_unquantized
sbatch evaluate_results.sh
```

## Output

Results will be saved to: `data/gen_res_prompt_ans_only_llama3_unquantized/`

## Notes

- The model is run in **unquantized** mode for maximum quality
- Uses vLLM for faster inference
- All experiments use the "near" scenario with random documents
- Gold document is always placed at the last position (near the query)



