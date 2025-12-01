import os 
import argparse
import warnings
import re
import glob
from tqdm import tqdm
from typing import Tuple, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from llm import LLM as LLM_HF
from llm_vllm import LLM as LLM_VLLM

# Type alias for LLM (can be either HuggingFace or vLLM backend)
LLM = Union[LLM_HF, LLM_VLLM]
from utils import *
from prompt_dataset import PromptDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED=10

info = {
    "data_path": 'data/10k_train_dataset.json',
    "random_results_path": "data/10k_random_results_at60.pkl",
    "adore_search_results_path": "data/adore_search_results_at200.pkl",
    "contriever_search_results_path": "data/contriever_search_results_at150.pkl",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM Generation.")
    parser.add_argument('--output_dir', type=str, default='data/gen_res', help='Output directory')
    parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length for the LLM model', default=4096)
    parser.add_argument('--load_full_corpus', type=str2bool, help='Load the full corpus', default=True)    
    parser.add_argument('--use_random', type=str2bool, help='Use random irrelevant documents')
    parser.add_argument('--use_adore', type=str2bool, help="Use the retrieved documents from ADORE", default=False)
    parser.add_argument('--gold_position', type=int, help='The (0-indexed) position of the gold document in the context')
    parser.add_argument('--num_documents_in_context', type=int, help='Total number of documents in the context')
    parser.add_argument('--get_documents_without_answer', type=str2bool, help='Select only documents without the answer (e.g., distracting)', default=True)
    parser.add_argument('--repeat_gold', type=str2bool, help='Repeat only the golden document num_documents_in_context times (no random documents)', default=False)
    parser.add_argument('--pad_gold', type=str2bool, help='Replace non-golden documents with padding tokens matching their size, place golden document at end', default=False)
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate', default=15)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_every', type=int, default=250)
    parser.add_argument('--use_vllm', type=str2bool, help='Use vLLM backend instead of HuggingFace', default=False)
    parser.add_argument('--quantization', type=str, help='Quantization method: "awq", "gptq", "fp8", or None. For HF: "4" or "8" for bitsandbytes', default=None)
    parser.add_argument('--tensor_parallel_size', type=int, help='Number of GPUs for tensor parallelism (vLLM only)', default=1)
    parser.add_argument('--dtype', type=str, help='Data type for model (vLLM only)', default='bfloat16')

    args = parser.parse_args()

    if args.num_documents_in_context is None:
        parser.error("'num_documents_in_context' must be specified.")
    if args.num_documents_in_context <= 0:
        parser.error("'num_documents_in_context' must be a positive integer.")
    if args.gold_position is not None and (args.gold_position < 0 or args.gold_position >= args.num_documents_in_context):
        parser.error("'gold_position' must be within the range of 'num_documents_in_context'.")

    return args


def load_corpus(
    args: argparse.Namespace
) -> Tuple[List[Dict], Optional[Dict[int, int]]]:
    # Load the corpus
    if args.load_full_corpus:
        corpus = read_corpus_json('data/corpus.json')
        return corpus, None

    if args.use_random:
        corpus, full_to_subset_idx_map = read_corpus_with_random()
    elif args.use_adore:
        corpus, full_to_subset_idx_map = read_corpus_with_adore()
    else: 
        # Corpus with documents from Contriever
        corpus, full_to_subset_idx_map = read_corpus_with_contriever()

    return corpus, full_to_subset_idx_map


def load_search_results(args: argparse.Namespace) -> List[Tuple[List[int], List[float]]]:
    # Decide on search results path based on conditions
    if args.use_random:
        search_results_path = info['random_results_path']
    elif args.use_adore:
        search_results_path = info['adore_search_results_path']
    else:
        # Search results from Contriever
        search_results_path = info['contriever_search_results_path'] 

    search_results = read_pickle(search_results_path)
    return search_results


def initialize_dataset_and_loader(
    args: argparse.Namespace, 
    corpus: List[Dict], 
    full_to_subset_idx_map: Optional[Dict[int, int]], 
    search_results: List[Tuple[List[int], List[float]]], 
    tokenizer: PreTrainedTokenizer
) -> DataLoader:
    
    prompt_ds = PromptDataset(
        corpus=corpus, data_path=info['data_path'], 
        tokenizer=tokenizer, 
        max_tokenized_length=args.model_max_length - 2, 
        search_results=search_results,
        full_to_subset_idx_map=full_to_subset_idx_map,
        do_normalize_query=True, 
        num_documents_in_context=args.num_documents_in_context,
        gold_position=args.gold_position,
        get_documents_without_answer=args.get_documents_without_answer,
        repeat_gold=args.repeat_gold,
        pad_gold=args.pad_gold,
    )
    prompt_dataloader = DataLoader(
        prompt_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return prompt_dataloader


def print_info(args: argparse.Namespace):
    print("INFO:")
    print(f"DATA: {info['data_path']}")
    print(f"MODEL: {args.llm_id}")
    print(f"BACKEND: {'vLLM' if args.use_vllm else 'HuggingFace'}")
    print(f"QUANTIZATION: {args.quantization or 'None'}")
    print(f"USE RANDOM IN CONTEXT: {args.use_random}")
    print(f"USE ADORE: {args.use_adore}")
    print(f"GOLD POSITION: {args.gold_position}")
    print(f"NUM DOCUMENTS IN CONTEXT: {args.num_documents_in_context}")
    print(f"DOCUMENTS WITHOUT ANSWER: {args.get_documents_without_answer}")
    print(f"REPEAT GOLD: {args.repeat_gold}")
    print(f"PAD GOLD: {args.pad_gold}")
    print(f"BATCH SIZE: {args.batch_size}")
    print(f"SAVE EVERY: {args.save_every}")


def generate_and_save(
    args: argparse.Namespace, 
    llm: LLM, 
    prompt_dataloader: DataLoader
):
    # Info from arguments
    llm_id = args.llm_id
    num_doc = args.num_documents_in_context
    save_every = args.save_every
    gold_pos = args.gold_position
    retriever_str = "adore" if args.use_adore else "contriever"
    rand_str = "_rand" if args.use_random else ""
    answerless_str = "_answerless" if args.get_documents_without_answer else ""
    repeat_gold_str = "_repeat_gold" if args.repeat_gold else ""
    pad_gold_str = "_pad_gold" if args.pad_gold else ""

    # Create the saving directory
    llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    saving_dir = f"{args.output_dir}/{llm_folder}/train/classic/{retriever_str}/{num_doc}_doc"
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    # TEMPORARY FIX: Check for existing checkpoints and skip already processed batches
    filename_prefix = f"numdoc{num_doc}_gold_at{gold_pos}{rand_str}{answerless_str}{repeat_gold_str}{pad_gold_str}_info_"
    pattern = re.compile(r'(\d+).pkl')
    existing_files = glob.glob(os.path.join(saving_dir, f"{filename_prefix}*.pkl"))
    skip_until = 0
    if existing_files:
        checkpoint_numbers = []
        for file in existing_files:
            match = pattern.search(os.path.basename(file))
            if match:
                checkpoint_numbers.append(int(match.group(1)))
        if checkpoint_numbers:
            skip_until = max(checkpoint_numbers)
            print(f"TEMPORARY FIX: Found existing checkpoint at batch {skip_until}. Skipping batches 0-{skip_until}...")
    
    # MPT has a different answer string in the prompt
    answer_string_in_prompt = "### Response:" if 'mpt' in llm_id else "Answer:"

    all_info = []  
    for idx, prompt_batch in enumerate(tqdm(prompt_dataloader)):
        # TEMPORARY FIX: Skip batches that were already processed
        if (idx + 1) <= skip_until:
            continue
        prompts = prompt_batch['prompt']
        # vLLM handles batching internally, so we can pass the full list
        if args.use_vllm:
            generated_output = llm.generate(prompts, max_new_tokens=args.max_new_tokens, temperature=0.0, repetition_penalty=1.1)
        else:
            generated_output = llm.generate(prompts, max_new_tokens=args.max_new_tokens)
        
        generated_answers = []
        for output in generated_output:
            # vLLM returns only the generated text (not the prompt)
            # So we should use the output directly, but clean it up
            response = output.strip()
            
            # Clean up common issues:
            # 1. Remove placeholder-like text (lines of underscores)
            lines = response.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                # Skip lines that are mostly underscores or placeholders
                if line and len(line) > 2:
                    # Check if line is mostly underscores
                    if line.count('_') / len(line) > 0.7:
                        continue
                    # Stop at common continuation markers
                    if 'extract' in line.lower() and ('token' in line.lower() or 'document' in line.lower()):
                        break
                    if line.startswith('Extract') or line.startswith('(extract'):
                        break
                    # Stop at new question markers
                    if line.startswith('Question:') or line.startswith('Documents:'):
                        break
                    cleaned_lines.append(line)
            
            response = ' '.join(cleaned_lines).strip()
            
            # If response is still mostly underscores or very short placeholder, treat as empty
            if response:
                if response.count('_') / len(response) > 0.5:
                    response = ""
                elif len(response) < 3:
                    response = ""
            
            generated_answers.append(response)

        prompt_batch['generated_answer'] = generated_answers
        all_info.append(prompt_batch)
        
        if (idx + 1) % save_every == 0 or (idx + 1) == len(prompt_dataloader):
            print(f"Saving at {idx + 1}...")
            file_name = f"{saving_dir}/numdoc{num_doc}_gold_at{gold_pos}{rand_str}{answerless_str}{repeat_gold_str}{pad_gold_str}_info_{idx+1}.pkl"
            write_pickle(all_info, file_name)
            all_info = []


def main():
    args = parse_arguments()

    print("Loading LLM...")
    llm_id = args.llm_id
    
    if args.use_vllm:
        # Use vLLM backend
        quantization = None
        if args.quantization and args.quantization.strip():
            quant_lower = args.quantization.lower().strip()
            if quant_lower in ['awq', 'gptq', 'fp8']:
                quantization = quant_lower
            else:
                print(f"Warning: Unknown quantization method '{args.quantization}' for vLLM, using None")
        llm = LLM_VLLM(
            model_id=llm_id,
            quantization=quantization,
            model_max_length=args.model_max_length,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
        )
        tokenizer = llm.tokenizer
    else:
        # Use HuggingFace backend
        quantization_bits = None
        if args.quantization:
            if args.quantization in ['4', '8']:
                quantization_bits = int(args.quantization)
            else:
                print(f"Warning: For HuggingFace backend, quantization must be '4' or '8', got '{args.quantization}'. Using None.")
        
        llm = LLM_HF(
            llm_id, device, quantization_bits=quantization_bits, 
            model_max_length=args.model_max_length
        )
        tokenizer = llm.tokenizer
    
    print("LLM loaded")


    print("Loading corpus and search results...")
    corpus, full_to_subset_idx_map = load_corpus(args)
    search_results = load_search_results(args)
    print("Corpus and search results loaded")


    print("Loading prompt dataset...")
    prompt_dataloader = initialize_dataset_and_loader(
        args, corpus, full_to_subset_idx_map, search_results, tokenizer
    )
    print("Prompt dataset loaded")

    print_info(args)
    generate_and_save(args, llm, prompt_dataloader)



if __name__ == "__main__":
    seed_everything(SEED)
    main()